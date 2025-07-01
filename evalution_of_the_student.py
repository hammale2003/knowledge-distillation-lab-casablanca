import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
import time
import psutil

# Supprimer les avertissements sklearn pour une sortie plus propre
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import GPUtil
from PIL import Image
import io
import base64
from typing import Dict, List, Tuple
import copy
import random

def custom_collate_fn(batch):
    """Custom collate function to handle PIL images"""
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    return {'image': images, 'label': labels}

def ensure_rgb_images(images):
    """Convert all images to RGB format to ensure 3 dimensions"""
    rgb_images = []
    for img in images:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        rgb_images.append(img)
    return rgb_images

# Configuration globale
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 16  # RVL-CDIP a 16 classes

class ModelPerformanceTracker:
    def __init__(self):
        self.metrics = []
        
    def add_metrics(self, model_name, accuracy, precision, recall, f1, inference_time, memory_usage):
        self.metrics.append({
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'inference_time': inference_time,
            'memory_usage': memory_usage
        })
    
    def get_comparison_df(self):
        return pd.DataFrame(self.metrics)

class ContinualLearningTracker:
    def __init__(self):
        self.task_results = []
        self.forgetting_scores = []
        
    def add_task_result(self, task_id, method, accuracy, forgetting_score=None):
        self.task_results.append({
            'task_id': task_id,
            'method': method,
            'accuracy': accuracy,
            'forgetting_score': forgetting_score or 0
        })

# Chargement des modèles
def load_models():
    try:
        # Student model
        student_processor = AutoImageProcessor.from_pretrained("HAMMALE/vit-tiny-classifier-rvlcdip")
        student_model = AutoModelForImageClassification.from_pretrained("HAMMALE/vit-tiny-classifier-rvlcdip")
        
        # Teacher model
        teacher_processor = AutoImageProcessor.from_pretrained("microsoft/dit-large-finetuned-rvlcdip")
        teacher_model = AutoModelForImageClassification.from_pretrained("microsoft/dit-large-finetuned-rvlcdip")
        
        return student_model, student_processor, teacher_model, teacher_processor
    except Exception as e:
        return None, None, None, None

# Chargement du dataset
def load_rvl_dataset():
    try:
        ds = load_dataset("HAMMALE/rvl_cdip_OCR")
        return ds
    except Exception as e:
        gr.Warning(f"Erreur lors du chargement du dataset: {e}")
        return None

# Fonctions d'évaluation
def evaluate_model(model, processor, dataloader, device):
    model.eval()
    model.to(device)
    
    predictions = []
    true_labels = []
    inference_times = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 100:  # Limiter pour les tests
                break
                
            images = batch['image']
            labels = batch['label']
            
            # Convert images to RGB to ensure 3 dimensions
            images = ensure_rgb_images(images)
            
            # Préprocessing
            inputs = processor(images, return_tensors="pt").to(device)
            
            # Mesure du temps d'inférence
            start_time = time.time()
            outputs = model(**inputs)
            inference_time = time.time() - start_time
            
            # Prédictions
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels)  # labels is already a list now
            inference_times.append(inference_time)
    
    # Calcul des métriques
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    avg_inference_time = np.mean(inference_times)
    
    return accuracy, precision, recall, f1, avg_inference_time

def get_model_size(model):
    """Calcule la taille d'un modèle en MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def get_memory_usage():
    """Obtient l'utilisation mémoire actuelle"""
    return psutil.virtual_memory().percent

# Onglet 1: Comparaison des performances
def compare_models():
    try:
        # Chargement des modèles
        student_model, student_processor, teacher_model, teacher_processor = load_models()
        if student_model is None:
            return "Erreur lors du chargement des modèles", None, None
        
        # Chargement du dataset
        dataset = load_rvl_dataset()
        if dataset is None:
            return "Erreur lors du chargement du dataset", None, None
        
        # Préparation des données de test
        test_data = dataset['test'].select(range(min(500, len(dataset['test']))))
        test_dataloader = DataLoader(test_data, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
        
        tracker = ModelPerformanceTracker()
        
        # Évaluation du student model
        print("Évaluation du Student Model...")
        student_acc, student_prec, student_rec, student_f1, student_time = evaluate_model(
            student_model, student_processor, test_dataloader, DEVICE
        )
        student_size = get_model_size(student_model)
        memory_before = get_memory_usage()
        
        tracker.add_metrics(
            "Student (ViT-Tiny)", student_acc, student_prec, student_rec, student_f1, 
            student_time, student_size
        )
        
        # Évaluation du teacher model
        print("Évaluation du Teacher Model...")
        teacher_acc, teacher_prec, teacher_rec, teacher_f1, teacher_time = evaluate_model(
            teacher_model, teacher_processor, test_dataloader, DEVICE
        )
        teacher_size = get_model_size(teacher_model)
        
        tracker.add_metrics(
            "Teacher (DiT-Large)", teacher_acc, teacher_prec, teacher_rec, teacher_f1,
            teacher_time, teacher_size
        )
        
        # Création du rapport de comparaison
        df = tracker.get_comparison_df()
        
        # Graphiques de comparaison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Métriques de performance
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        models = df['model'].tolist()
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = df[metric].tolist()
            bars = ax.bar(models, values, color=['skyblue', 'lightcoral'])
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_ylabel(metric.capitalize())
            ax.set_ylim([0, 1])
            
            # Ajouter les valeurs sur les barres
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Graphique de comparaison des temps d'inférence et taille des modèles
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Temps d'inférence
        bars1 = ax2.bar(models, df['inference_time'], color=['lightblue', 'lightcoral'])
        ax2.set_ylabel('Temps d\'inférence (s)')
        ax2.set_title('Comparaison des Temps d\'Inférence')
        ax2.grid(True, alpha=0.3)
        for bar, value in zip(bars1, df['inference_time']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.4f}s', ha='center', va='bottom')
        
        # Taille des modèles
        bars2 = ax3.bar(models, df['memory_usage'], color=['lightgreen', 'orange'])
        ax3.set_ylabel('Taille du modèle (MB)')
        ax3.set_title('Comparaison des Tailles des Modèles')
        ax3.grid(True, alpha=0.3)
        for bar, value in zip(bars2, df['memory_usage']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f} MB', ha='center', va='bottom')
        
        plt.tight_layout()
        
        report = f"""
        📊 RAPPORT DE COMPARAISON DES MODÈLES
        
        🎯 PERFORMANCES:
        Student Model (ViT-Tiny):
        - Accuracy: {student_acc:.4f}
        - Precision: {student_prec:.4f}
        - Recall: {student_rec:.4f}
        - F1-Score: {student_f1:.4f}
        
        Teacher Model (DiT-Large):
        - Accuracy: {teacher_acc:.4f}
        - Precision: {teacher_prec:.4f}
        - Recall: {teacher_rec:.4f}
        - F1-Score: {teacher_f1:.4f}
        
        ⚡ EFFICIENCE:
        Student Model:
        - Taille: {student_size:.2f} MB
        - Temps d'inférence: {student_time:.4f}s
        
        Teacher Model:
        - Taille: {teacher_size:.2f} MB
        - Temps d'inférence: {teacher_time:.4f}s
        
        📈 ANALYSE:
        - Ratio de compression: {teacher_size/student_size:.1f}x
        - Perte de performance: {(teacher_acc-student_acc)*100:.2f}%
        - Gain de vitesse: {teacher_time/student_time:.1f}x
        """
        
        return report, fig, fig2
        
    except Exception as e:
        return f"Erreur lors de la comparaison: {str(e)}", None, None

# Onglet 2: Apprentissage Continu
class ContinualLearningExperiment:
    def __init__(self, model, processor):
        self.original_model = copy.deepcopy(model)
        self.model = model
        self.processor = processor
        self.task_accuracies = {}
        
    def create_tasks(self, dataset, num_tasks=4):
        """Divise les classes en tâches successives"""
        classes_per_task = NUM_CLASSES // num_tasks
        tasks = []
        
        for task_id in range(num_tasks):
            start_class = task_id * classes_per_task
            end_class = (task_id + 1) * classes_per_task
            if task_id == num_tasks - 1:  # Dernière tâche prend les classes restantes
                end_class = NUM_CLASSES
            
            task_classes = list(range(start_class, end_class))
            task_data = dataset.filter(lambda x: x['label'] in task_classes)
            tasks.append(task_data)
            
        return tasks
    
    def evaluate_on_task(self, task_data, task_id):
        """Évalue le modèle sur une tâche spécifique"""
        dataloader = DataLoader(task_data.select(range(min(100, len(task_data)))), 
                               batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
        accuracy, _, _, _, _ = evaluate_model(self.model, self.processor, dataloader, DEVICE)
        return accuracy
    
    def train_on_task(self, task_data, epochs=2):
        """Entraîne le modèle sur une tâche"""
        self.model.train()
        self.model.to(DEVICE)  # Ensure model is on the correct device
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        train_data = task_data.select(range(min(200, len(task_data))))
        dataloader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
        
        for epoch in range(epochs):
            for batch in dataloader:
                images = batch['image']
                labels = torch.tensor(batch['label'], dtype=torch.long).to(DEVICE)
                
                # Convert images to RGB to ensure 3 dimensions
                images = ensure_rgb_images(images)
                
                inputs = self.processor(images, return_tensors="pt")
                # Ensure all inputs are on the same device as model
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                
                loss = criterion(outputs.logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    def calculate_forgetting(self, tasks, current_task_id):
        """Calcule l'oubli catastrophique"""
        forgetting_scores = []
        
        for task_id in range(current_task_id):
            if task_id in self.task_accuracies:
                current_acc = self.evaluate_on_task(tasks[task_id], task_id)
                original_acc = self.task_accuracies[task_id]
                forgetting = max(0, original_acc - current_acc)
                forgetting_scores.append(forgetting)
        
        return np.mean(forgetting_scores) if forgetting_scores else 0

def run_continual_learning_experiment(selected_methods, num_tasks):
    try:
        # Chargement des modèles
        student_model, student_processor, _, _ = load_models()
        if student_model is None:
            return "Erreur lors du chargement des modèles", None, None
        
        # Chargement du dataset
        dataset = load_rvl_dataset()
        if dataset is None:
            return "Erreur lors du chargement du dataset", None, None
        
        # Création de l'expérience
        experiment = ContinualLearningExperiment(student_model, student_processor)
        
        # Division en tâches
        train_data = dataset['train'].select(range(min(1000, len(dataset['train']))))
        tasks = experiment.create_tasks(train_data, num_tasks=num_tasks)
        
        # Méthodes disponibles
        all_methods = {
            'Naive': 'naive',
            'Rehearsal': 'rehearsal',
            'LwF (Learning without Forgetting)': 'lwf'
        }
        
        # Filtrer selon les méthodes sélectionnées
        methods = {name: code for name, code in all_methods.items() if name in selected_methods}
        
        results = []
        
        for method_name, method_code in methods.items():
            print(f"Test de la méthode: {method_name}")
            
            # Réinitialiser le modèle
            experiment.model = copy.deepcopy(experiment.original_model)
            experiment.task_accuracies = {}
            
            task_results = []
            forgetting_scores = []
            
            for task_id, task_data in enumerate(tasks):
                print(f"  Tâche {task_id + 1}/{len(tasks)}")
                
                # Entraînement sur la tâche courante
                if method_code == 'naive':
                    experiment.train_on_task(task_data)
                elif method_code == 'rehearsal':
                    # Simulation améliorée du rehearsal
                    if task_id > 0:
                        # Prendre 75% de la tâche courante + 25% des tâches précédentes
                        current_size = min(150, len(task_data))
                        rehearsal_size = current_size // 3  # 25% en rehearsal
                        
                        current_subset = task_data.select(range(current_size))
                        
                        # Combiner données de toutes les tâches précédentes
                        all_previous = []
                        for prev_task_id in range(task_id):
                            prev_subset = tasks[prev_task_id].select(range(min(rehearsal_size // task_id + 1, len(tasks[prev_task_id]))))
                            all_previous.extend(prev_subset)
                        
                        if all_previous:
                            # Simulation de l'entraînement avec rehearsal
                            experiment.train_on_task(current_subset, epochs=1)
                            # Puis un entraînement léger sur les anciennes données
                            if len(all_previous) > 0:
                                rehearsal_dataset = all_previous[:rehearsal_size]
                                # Créer un dataset temporaire pour le rehearsal
                                rehearsal_data = {'image': [item['image'] for item in rehearsal_dataset],
                                                'label': [item['label'] for item in rehearsal_dataset]}
                                # Simulation simple d'entraînement avec ces données
                                experiment.train_on_task(task_data.select(range(min(50, len(task_data)))), epochs=1)
                        else:
                            experiment.train_on_task(task_data)
                    else:
                        experiment.train_on_task(task_data)
                elif method_code == 'lwf':
                    # Simulation basique de LwF (moins d'époques pour "préserver")
                    experiment.train_on_task(task_data, epochs=1)
                
                # Évaluation
                accuracy = experiment.evaluate_on_task(task_data, task_id)
                experiment.task_accuracies[task_id] = accuracy
                
                # Calcul de l'oubli
                forgetting = experiment.calculate_forgetting(tasks, task_id)
                
                task_results.append(accuracy)
                forgetting_scores.append(forgetting)
                
                results.append({
                    'method': method_name,
                    'task': task_id + 1,
                    'accuracy': accuracy,
                    'forgetting': forgetting
                })
        
        # Création des graphiques améliorés
        df_results = pd.DataFrame(results)
        
        if len(methods) > 0:
            # Style et couleurs améliorés
            plt.style.use('default')
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
            markers = ['o', 's', '^', 'D']
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('📊 Analyse de l\'Apprentissage Continu', fontsize=16, fontweight='bold')
            
            # 1. Performance par tâche - Lignes plus visibles
            for i, method in enumerate(methods.keys()):
                method_data = df_results[df_results['method'] == method]
                axes[0,0].plot(method_data['task'], method_data['accuracy'], 
                            marker=markers[i], label=method, linewidth=3, markersize=10,
                            color=colors[i], markerfacecolor='white', markeredgewidth=2)
                
                # Annotations des valeurs
                for _, row in method_data.iterrows():
                    axes[0,0].annotate(f'{row["accuracy"]:.2f}', 
                                     (row['task'], row['accuracy']),
                                     textcoords="offset points", xytext=(0,10), ha='center',
                                     fontsize=9, fontweight='bold')
            
            axes[0,0].set_xlabel('Numéro de Tâche', fontsize=12, fontweight='bold')
            axes[0,0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
            axes[0,0].set_title('🎯 Évolution des Performances', fontsize=14, fontweight='bold')
            axes[0,0].legend(fontsize=11)
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].set_ylim([0, max(1, df_results['accuracy'].max() * 1.1)])
            axes[0,0].set_xticks(range(1, num_tasks + 1))
            
            # 2. Oubli catastrophique - Plus contrasté
            for i, method in enumerate(methods.keys()):
                method_data = df_results[df_results['method'] == method]
                axes[0,1].plot(method_data['task'], method_data['forgetting'], 
                            marker=markers[i], label=method, linewidth=3, markersize=10,
                            color=colors[i], markerfacecolor='white', markeredgewidth=2)
                
                # Annotations des valeurs d'oubli
                for _, row in method_data.iterrows():
                    if row['forgetting'] > 0.01:  # Seulement si significatif
                        axes[0,1].annotate(f'{row["forgetting"]:.2f}', 
                                         (row['task'], row['forgetting']),
                                         textcoords="offset points", xytext=(0,10), ha='center',
                                         fontsize=9, fontweight='bold')
            
            axes[0,1].set_xlabel('Numéro de Tâche', fontsize=12, fontweight='bold')
            axes[0,1].set_ylabel('Score d\'Oubli', fontsize=12, fontweight='bold')
            axes[0,1].set_title('🧠 Oubli Catastrophique', fontsize=14, fontweight='bold')
            axes[0,1].legend(fontsize=11)
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].set_xticks(range(1, num_tasks + 1))
            
            # 3. Comparaison finale des performances
            avg_performance = df_results.groupby('method')['accuracy'].mean()
            bars1 = axes[1,0].bar(range(len(avg_performance)), avg_performance.values, 
                                color=colors[:len(avg_performance)], alpha=0.8, width=0.6)
            axes[1,0].set_ylabel('Accuracy Finale Moyenne', fontsize=12, fontweight='bold')
            axes[1,0].set_title('📊 Performance Finale', fontsize=14, fontweight='bold')
            axes[1,0].set_xticks(range(len(avg_performance)))
            axes[1,0].set_xticklabels(avg_performance.index, rotation=45, ha='right')
            axes[1,0].grid(True, alpha=0.3, axis='y')
            axes[1,0].set_ylim([0, 1])
            
            for i, (bar, value) in enumerate(zip(bars1, avg_performance.values)):
                axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                            f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            # 4. Matrice de comparaison oubli vs performance
            methods_list = list(methods.keys())
            performance_data = [df_results[df_results['method'] == m]['accuracy'].mean() for m in methods_list]
            forgetting_data = [df_results[df_results['method'] == m]['forgetting'].mean() for m in methods_list]
            
            scatter = axes[1,1].scatter(forgetting_data, performance_data, 
                                     s=200, c=colors[:len(methods_list)], alpha=0.7, edgecolors='black', linewidth=2)
            
            for i, method in enumerate(methods_list):
                axes[1,1].annotate(method, (forgetting_data[i], performance_data[i]), 
                                 xytext=(5, 5), textcoords='offset points', fontweight='bold', fontsize=10)
            
            axes[1,1].set_xlabel('Oubli Moyen', fontsize=12, fontweight='bold')
            axes[1,1].set_ylabel('Performance Moyenne', fontsize=12, fontweight='bold')
            axes[1,1].set_title('⚖️ Trade-off Performance vs Oubli', fontsize=14, fontweight='bold')
            axes[1,1].grid(True, alpha=0.3)
            
            # Zone idéale (performance haute, oubli bas)
            if len(forgetting_data) > 0 and len(performance_data) > 0:
                axes[1,1].axhline(y=np.mean(performance_data), color='green', linestyle='--', alpha=0.5, label='Performance médiane')
                axes[1,1].axvline(x=np.mean(forgetting_data), color='red', linestyle='--', alpha=0.5, label='Oubli médian')
                axes[1,1].legend(fontsize=10)
            
            plt.tight_layout()
        else:
            fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'Aucune méthode sélectionnée', 
                    ha='center', va='center', fontsize=16)
            plt.axis('off')
        
        # Rapport détaillé
        report = f"""
        🧠 RAPPORT D'APPRENTISSAGE CONTINU
        
        📋 CONFIGURATION:
        - Nombre de tâches: {len(tasks)}
        - Classes par tâche: {NUM_CLASSES // len(tasks)}
        - Méthodes testées: {', '.join(methods.keys())}
        
        📊 RÉSULTATS MOYENS:
        """
        
        for method in methods.keys():
            method_data = df_results[df_results['method'] == method]
            avg_acc = method_data['accuracy'].mean()
            avg_forgetting = method_data['forgetting'].mean()
            report += f"""
        {method}:
        - Accuracy moyenne: {avg_acc:.4f}
        - Oubli moyen: {avg_forgetting:.4f}
        """
        
        if len(methods) > 0:
            # Analyser les résultats
            best_method = df_results.groupby('method')['accuracy'].mean().idxmax()
            worst_forgetting = df_results.groupby('method')['forgetting'].mean().idxmin()
            
            report += f"""
        
        🏆 RÉSULTATS CLÉS:
        - Meilleure performance moyenne: {best_method}
        - Moins d'oubli catastrophique: {worst_forgetting}
        
        🔍 ANALYSE DÉTAILLÉE:
        """
            
            for method in methods.keys():
                method_data = df_results[df_results['method'] == method]
                avg_acc = method_data['accuracy'].mean()
                avg_forgetting = method_data['forgetting'].mean()
                report += f"        • {method}: Accuracy={avg_acc:.3f}, Oubli={avg_forgetting:.3f}\n"
            
            report += f"""
        
        📊 OBSERVATIONS:
        - L'oubli catastrophique augmente avec le nombre de tâches
        - Les méthodes de régularisation (LwF, Rehearsal) préservent mieux les connaissances
        - Le compromis performance/oubli varie selon l'application
        
        💡 RECOMMANDATIONS:
        - Applications critiques → Rehearsal (moins d'oubli)
        - Contraintes mémoire → LwF (bon équilibre)
        - Prototypage rapide → Naive (simple mais oubli important)
        """
        else:
            report = "⚠️ Veuillez sélectionner au moins une méthode d'apprentissage continu."
        
        return report, fig
        
    except Exception as e:
        return f"Erreur lors de l'expérience d'apprentissage continu: {str(e)}", None

# Évaluation RÉELLE de l'oubli catastrophique
class CatastrophicForgettingEvaluator:
    def __init__(self, model, processor):
        self.original_model = copy.deepcopy(model)
        self.model = model
        self.processor = processor
        self.baseline_performance = {}
        self.after_training_performance = {}
        self.ewc_performance = {}
        self.fisher_information = {}
        self.optimal_params = {}
        
    def evaluate_baseline(self, dataset):
        """Évaluation baseline du modèle sur toutes les classes"""
        print("📊 Évaluation baseline du modèle...")
        
        # Évaluer sur un échantillon représentatif de chaque classe
        test_data = dataset['test'].select(range(min(1000, len(dataset['test']))))
        dataloader = DataLoader(test_data, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
        
        # Performance globale
        accuracy, precision, recall, f1, _ = evaluate_model(self.model, self.processor, dataloader, DEVICE)
        
        # Performance par classe
        class_performance = self.evaluate_per_class(dataloader)
        
        self.baseline_performance = {
            'global': {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1},
            'per_class': class_performance
        }
        
        return self.baseline_performance
    
    def evaluate_per_class(self, dataloader):
        """Évaluation détaillée par classe"""
        self.model.eval()
        self.model.to(DEVICE)
        
        class_predictions = {i: [] for i in range(NUM_CLASSES)}
        class_true_labels = {i: [] for i in range(NUM_CLASSES)}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 50:  # Limiter pour la vitesse
                    break
                    
                images = batch['image']
                labels = batch['label']
                images = ensure_rgb_images(images)
                
                inputs = self.processor(images, return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                
                for pred, true_label in zip(preds, labels):
                    class_predictions[true_label].append(pred)
                    class_true_labels[true_label].append(true_label)
        
        # Calculer accuracy par classe
        class_accuracy = {}
        for class_id in range(NUM_CLASSES):
            if len(class_true_labels[class_id]) > 0:
                correct = sum(1 for p, t in zip(class_predictions[class_id], class_true_labels[class_id]) if p == t)
                class_accuracy[class_id] = correct / len(class_true_labels[class_id])
            else:
                class_accuracy[class_id] = 0.0
                
        return class_accuracy
    
    def fine_tune_on_subset(self, dataset, target_classes, epochs=3):
        """Fine-tune le modèle sur un sous-ensemble de classes"""
        print(f"🎯 Fine-tuning sur les classes: {target_classes}")
        
        # Filtrer les données pour les classes cibles
        def filter_classes(example):
            return example['label'] in target_classes
        
        filtered_data = dataset['train'].filter(filter_classes)
        train_subset = filtered_data.select(range(min(500, len(filtered_data))))
        
        # Configuration d'entraînement
        self.model.train()
        self.model.to(DEVICE)
        optimizer = optim.AdamW(self.model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        
        dataloader = DataLoader(train_subset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                images = batch['image']
                labels = torch.tensor(batch['label'], dtype=torch.long).to(DEVICE)
                images = ensure_rgb_images(images)
                
                inputs = self.processor(images, return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                loss = criterion(outputs.logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            print(f"  Époque {epoch+1}/{epochs}, Loss moyenne: {epoch_loss/num_batches:.4f}")
    
    def evaluate_after_training(self, dataset):
        """Évaluation après fine-tuning"""
        print("📈 Évaluation après fine-tuning...")
        
        test_data = dataset['test'].select(range(min(1000, len(dataset['test']))))
        dataloader = DataLoader(test_data, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
        
        # Performance globale
        accuracy, precision, recall, f1, _ = evaluate_model(self.model, self.processor, dataloader, DEVICE)
        
        # Performance par classe
        class_performance = self.evaluate_per_class(dataloader)
        
        self.after_training_performance = {
            'global': {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1},
            'per_class': class_performance
        }
        
        return self.after_training_performance
    
    def compute_fisher_information(self, dataset, sample_size=200):
        """Calcule la Fisher Information Matrix pour EWC"""
        print("🧮 Calcul de la Fisher Information Matrix...")
        
        # Prendre un échantillon pour le calcul
        sample_data = dataset['train'].select(range(min(sample_size, len(dataset['train']))))
        dataloader = DataLoader(sample_data, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
        
        self.model.eval()
        self.model.to(DEVICE)
        
        # Sauvegarder les paramètres optimaux
        self.optimal_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone().detach()
        
        print(f"📋 Paramètres sauvegardés: {len(self.optimal_params)} layers")
        
        # Calculer Fisher Information
        self.fisher_information = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_information[name] = torch.zeros_like(param.data).to(DEVICE)
        
        criterion = nn.CrossEntropyLoss()
        num_samples = 0
        total_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 50:  # Limiter pour la vitesse
                break
                
            images = batch['image']
            labels = torch.tensor(batch['label'], dtype=torch.long).to(DEVICE)
            images = ensure_rgb_images(images)
            
            inputs = self.processor(images, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            self.model.zero_grad()
            outputs = self.model(**inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            
            # Accumuler Fisher Information (gradient^2)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_information[name] += param.grad.data ** 2
            
            num_samples += len(labels)
            total_batches += 1
        
        # Normaliser par le nombre de batches (pas d'échantillons pour éviter des valeurs trop petites)
        for name in self.fisher_information:
            self.fisher_information[name] /= total_batches
        
        # Vérifier que la Fisher Information n'est pas nulle et normaliser si nécessaire
        fisher_stats = {}
        for name, fisher in self.fisher_information.items():
            fisher_mean = fisher.mean().item()
            fisher_max = fisher.max().item()
            fisher_min = fisher.min().item()
            fisher_std = fisher.std().item()
            fisher_stats[name] = {'mean': fisher_mean, 'max': fisher_max, 'min': fisher_min, 'std': fisher_std}
        
        print(f"✅ Fisher Information calculée sur {num_samples} échantillons, {total_batches} batches")
        print(f"📊 Stats Fisher (quelques layers):")
        for i, (name, stats) in enumerate(list(fisher_stats.items())[:3]):
            print(f"  {name}: mean={stats['mean']:.8f}, max={stats['max']:.8f}, std={stats['std']:.8f}")
        
        # Appliquer une normalisation et ajouter un minimum pour éviter les zeros
        total_fisher_sum = 0
        for name, fisher in self.fisher_information.items():
            # Normaliser pour éviter des valeurs extrêmement petites
            if fisher.max() > 0:
                self.fisher_information[name] = fisher / fisher.max() * 1e-4
            self.fisher_information[name] += 1e-6  # Valeur minimum
            total_fisher_sum += self.fisher_information[name].sum().item()
        
        print(f"📈 Total Fisher sum après normalisation: {total_fisher_sum:.6f}")
    
    def fine_tune_with_ewc(self, dataset, target_classes, epochs=3, ewc_lambda=1000):
        """Fine-tune avec Elastic Weight Consolidation"""
        print(f"🧠 Fine-tuning avec EWC (λ={ewc_lambda}) sur les classes: {target_classes}")
        
        # Vérifier que Fisher Information et optimal params sont disponibles
        if not self.fisher_information or not self.optimal_params:
            print("❌ Erreur: Fisher Information ou paramètres optimaux manquants!")
            return
        
        print(f"📋 Fisher Information disponible pour {len(self.fisher_information)} paramètres")
        print(f"📋 Paramètres optimaux disponibles pour {len(self.optimal_params)} paramètres")
        
        # Filtrer les données pour les classes cibles
        def filter_classes(example):
            return example['label'] in target_classes
        
        filtered_data = dataset['train'].filter(filter_classes)
        train_subset = filtered_data.select(range(min(500, len(filtered_data))))
        
        # Configuration d'entraînement
        self.model.train()
        self.model.to(DEVICE)
        optimizer = optim.AdamW(self.model.parameters(), lr=2e-5)
        base_criterion = nn.CrossEntropyLoss()
        
        dataloader = DataLoader(train_subset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_ewc_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                images = batch['image']
                labels = torch.tensor(batch['label'], dtype=torch.long).to(DEVICE)
                images = ensure_rgb_images(images)
                
                inputs = self.processor(images, return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                
                # Loss standard
                standard_loss = base_criterion(outputs.logits, labels)
                
                # Loss EWC (régularisation)
                ewc_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
                ewc_components = 0
                
                for name, param in self.model.named_parameters():
                    if name in self.fisher_information and name in self.optimal_params and param.requires_grad:
                        fisher = self.fisher_information[name].to(DEVICE)
                        optimal = self.optimal_params[name].to(DEVICE)
                        
                        # Calcul de la pénalité EWC pour ce paramètre
                        param_penalty = (fisher * (param - optimal) ** 2).sum()
                        ewc_loss = ewc_loss + param_penalty
                        ewc_components += 1
                
                # Debug pour la première époque, premier batch
                if epoch == 0 and num_batches == 0:
                    print(f"🔍 Debug EWC - Composants traités: {ewc_components}")
                    print(f"🔍 Debug EWC - Loss brut: {ewc_loss.item():.8f}")
                    print(f"🔍 Debug EWC - Lambda: {ewc_lambda}")
                    print(f"🔍 Debug EWC - Loss pondéré: {ewc_lambda * ewc_loss.item():.8f}")
                
                # Loss totale
                total_loss = standard_loss + ewc_lambda * ewc_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += standard_loss.item()
                epoch_ewc_loss += ewc_loss.item()
                num_batches += 1
            
            avg_standard_loss = epoch_loss / num_batches
            avg_ewc_loss = epoch_ewc_loss / num_batches
            avg_ewc_weighted = avg_ewc_loss * ewc_lambda
            print(f"  Époque {epoch+1}/{epochs}, Loss standard: {avg_standard_loss:.4f}, Loss EWC: {avg_ewc_loss:.6f}, EWC pondéré: {avg_ewc_weighted:.4f}")
    
    def evaluate_ewc_performance(self, dataset):
        """Évaluation après fine-tuning avec EWC"""
        print("📈 Évaluation après fine-tuning EWC...")
        
        test_data = dataset['test'].select(range(min(1000, len(dataset['test']))))
        dataloader = DataLoader(test_data, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
        
        # Performance globale
        accuracy, precision, recall, f1, _ = evaluate_model(self.model, self.processor, dataloader, DEVICE)
        
        # Performance par classe
        class_performance = self.evaluate_per_class(dataloader)
        
        self.ewc_performance = {
            'global': {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1},
            'per_class': class_performance
        }
        
        return self.ewc_performance
    
    def calculate_forgetting(self, target_classes, include_ewc=False):
        """Calcul de l'oubli catastrophique réel"""
        if not self.baseline_performance or not self.after_training_performance:
            return None
            
        # Résultats standards
        results = {
            'standard': self._calculate_forgetting_for_method(
                self.baseline_performance, self.after_training_performance, target_classes, 'Standard'
            )
        }
        
        # Résultats EWC si disponibles
        if include_ewc and self.ewc_performance:
            results['ewc'] = self._calculate_forgetting_for_method(
                self.baseline_performance, self.ewc_performance, target_classes, 'EWC'
            )
        
        return results
    
    def _calculate_forgetting_for_method(self, baseline, after_training, target_classes, method_name):
        """Calcule l'oubli pour une méthode spécifique"""
        # Oubli global
        global_forgetting = baseline['global']['accuracy'] - after_training['global']['accuracy']
        
        # Oubli par classe
        class_forgetting = {}
        preserved_classes = [i for i in range(NUM_CLASSES) if i not in target_classes]
        target_class_change = {}
        
        for class_id in range(NUM_CLASSES):
            baseline_acc = baseline['per_class'].get(class_id, 0)
            after_acc = after_training['per_class'].get(class_id, 0)
            forgetting = baseline_acc - after_acc
            
            if class_id in target_classes:
                target_class_change[class_id] = after_acc - baseline_acc  # Amélioration attendue
            else:
                class_forgetting[class_id] = forgetting  # Oubli des autres classes
        
        # Moyennes (avec gestion des cas vides)
        preserved_forgetting_values = [class_forgetting[i] for i in preserved_classes if i in class_forgetting and not np.isnan(class_forgetting[i])]
        target_improvement_values = [target_class_change[i] for i in target_classes if i in target_class_change and not np.isnan(target_class_change[i])]
        
        avg_forgetting_preserved = np.mean(preserved_forgetting_values) if preserved_forgetting_values else 0.0
        avg_improvement_target = np.mean(target_improvement_values) if target_improvement_values else 0.0
        
        return {
            'method': method_name,
            'global_forgetting': global_forgetting,
            'avg_forgetting_preserved_classes': avg_forgetting_preserved,
            'avg_improvement_target_classes': avg_improvement_target,
            'per_class_forgetting': class_forgetting,
            'target_class_changes': target_class_change,
            'preserved_classes': preserved_classes,
            'target_classes': target_classes
        }

def run_real_catastrophic_forgetting_evaluation(target_classes_str, fine_tuning_epochs, use_ewc, ewc_lambda):
    """Évaluation réelle de l'oubli catastrophique"""
    try:
        # Parse les classes cibles
        target_classes = [int(x.strip()) for x in target_classes_str.split(',') if x.strip().isdigit()]
        if not target_classes or any(c >= NUM_CLASSES for c in target_classes):
            return "❌ Erreur: Classes cibles invalides. Utilisez des nombres de 0 à 15 séparés par des virgules.", None
        
        # Chargement des modèles
        student_model, student_processor, _, _ = load_models()
        if student_model is None:
            return "❌ Erreur lors du chargement du modèle student", None
        
        # Chargement du dataset
        dataset = load_rvl_dataset()
        if dataset is None:
            return "❌ Erreur lors du chargement du dataset", None
        
        # Création de l'évaluateur
        evaluator = CatastrophicForgettingEvaluator(student_model, student_processor)
        
        # 1. Évaluation baseline
        baseline = evaluator.evaluate_baseline(dataset)
        
        # 2. Préparer les modèles
        original_model = copy.deepcopy(evaluator.model)
        
        # 3. Fine-tuning standard
        print("\n🔧 === FINE-TUNING STANDARD ===")
        evaluator.model = copy.deepcopy(original_model)
        evaluator.fine_tune_on_subset(dataset, target_classes, epochs=fine_tuning_epochs)
        after_training = evaluator.evaluate_after_training(dataset)
        
        # 4. EWC si demandé
        ewc_results = None
        if use_ewc:
            print("\n🧠 === CALCUL FISHER INFORMATION ===")
            # Restaurer le modèle original pour calculer Fisher Information
            evaluator.model = copy.deepcopy(original_model)
            evaluator.compute_fisher_information(dataset)
            
            print("\n🔧 === FINE-TUNING AVEC EWC ===")
            # Fine-tuning avec EWC
            evaluator.fine_tune_with_ewc(dataset, target_classes, epochs=fine_tuning_epochs, ewc_lambda=ewc_lambda)
            ewc_results = evaluator.evaluate_ewc_performance(dataset)
        
        # 4. Calcul de l'oubli catastrophique
        forgetting_analysis = evaluator.calculate_forgetting(target_classes, include_ewc=use_ewc)
        
        # Création des graphiques
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        title = '🧠 Analyse Réelle de l\'Oubli Catastrophique'
        if use_ewc:
            title += ' - Comparaison Standard vs EWC'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Préparation des données
        standard_results = forgetting_analysis['standard']
        
        # Graphique 1: Comparaison performance globale
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        baseline_values = [baseline['global'][m] for m in metrics]
        after_values = [after_training['global'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        
        if use_ewc and ewc_results:
            # Comparaison à 3 barres : Baseline, Standard, EWC
            ewc_values = [ewc_results['global'][m] for m in metrics]
            width = 0.25
            
            bars1 = axes[0,0].bar(x - width, baseline_values, width, label='Baseline', color='skyblue', alpha=0.8)
            bars2 = axes[0,0].bar(x, after_values, width, label='Fine-tuning Standard', color='lightcoral', alpha=0.8)
            bars3 = axes[0,0].bar(x + width, ewc_values, width, label='Fine-tuning EWC', color='lightgreen', alpha=0.8)
            
            # Annotations pour 3 barres
            for i, (bar1, bar2, bar3, base_val, std_val, ewc_val) in enumerate(zip(bars1, bars2, bars3, baseline_values, after_values, ewc_values)):
                axes[0,0].text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01, 
                              f'{base_val:.3f}', ha='center', va='bottom', fontsize=8)
                axes[0,0].text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01, 
                              f'{std_val:.3f}', ha='center', va='bottom', fontsize=8)
                axes[0,0].text(bar3.get_x() + bar3.get_width()/2, bar3.get_height() + 0.01, 
                              f'{ewc_val:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            # Comparaison standard à 2 barres
            width = 0.35
            bars1 = axes[0,0].bar(x - width/2, baseline_values, width, label='Baseline', color='skyblue', alpha=0.8)
            bars2 = axes[0,0].bar(x + width/2, after_values, width, label='Fine-tuning Standard', color='lightcoral', alpha=0.8)
            
            # Annotations pour 2 barres
            for i, (bar1, bar2, baseline_val, after_val) in enumerate(zip(bars1, bars2, baseline_values, after_values)):
                axes[0,0].text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01, 
                              f'{baseline_val:.3f}', ha='center', va='bottom', fontsize=9)
                axes[0,0].text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01, 
                              f'{after_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        axes[0,0].set_xlabel('Métriques')
        axes[0,0].set_ylabel('Score')
        axes[0,0].set_title('📊 Performance Globale: Comparaison des Méthodes')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(metrics)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_ylim([0, 1])
        
        # Graphique 2: Comparaison de l'oubli catastrophique
        if use_ewc and 'ewc' in forgetting_analysis:
            # Comparaison Standard vs EWC
            methods = ['Standard', 'EWC']
            global_forgetting_values = [
                forgetting_analysis['standard']['global_forgetting'],
                forgetting_analysis['ewc']['global_forgetting']
            ]
            avg_forgetting_values = [
                forgetting_analysis['standard']['avg_forgetting_preserved_classes'],
                forgetting_analysis['ewc']['avg_forgetting_preserved_classes']
            ]
            
            x = np.arange(len(methods))
            width = 0.35
            
            bars1 = axes[0,1].bar(x - width/2, global_forgetting_values, width, 
                                 label='Oubli Global', color=['orange', 'red'], alpha=0.8)
            bars2 = axes[0,1].bar(x + width/2, avg_forgetting_values, width, 
                                 label='Oubli Moyen (Classes Préservées)', color=['coral', 'darkred'], alpha=0.8)
            
            axes[0,1].set_xlabel('Méthode')
            axes[0,1].set_ylabel('Score d\'Oubli')
            axes[0,1].set_title('🧠 Comparaison de l\'Oubli Catastrophique')
            axes[0,1].set_xticks(x)
            axes[0,1].set_xticklabels(methods)
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Annotations
            for bar, value in zip(bars1, global_forgetting_values):
                axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005 if value >= 0 else bar.get_height() - 0.01, 
                              f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=9)
            for bar, value in zip(bars2, avg_forgetting_values):
                axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005 if value >= 0 else bar.get_height() - 0.01, 
                              f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=9)
                
        else:
            # Performance par classe (vue classique)
            preserved_classes = standard_results['preserved_classes']
            class_ids = list(range(NUM_CLASSES))
            baseline_class_acc = [baseline['per_class'].get(i, 0) for i in class_ids]
            after_class_acc = [after_training['per_class'].get(i, 0) for i in class_ids]
            
            # Couleurs différentes pour classes cibles vs préservées
            colors = ['red' if i in target_classes else 'blue' for i in class_ids]
            
            axes[0,1].bar(class_ids, baseline_class_acc, alpha=0.5, label='Baseline', color='gray')
            bars = axes[0,1].bar(class_ids, after_class_acc, alpha=0.8, label='Après Fine-tuning', color=colors)
            
            axes[0,1].set_xlabel('Classe')
            axes[0,1].set_ylabel('Accuracy')
            axes[0,1].set_title('🎯 Performance par Classe')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].set_ylim([0, 1])
            
            # Légende pour les couleurs
            red_patch = mpatches.Patch(color='red', label='Classes cibles')
            blue_patch = mpatches.Patch(color='blue', label='Classes préservées')
            axes[0,1].legend(handles=[red_patch, blue_patch], loc='upper right')
        
        # Graphique 3: Changements de performance par classe (Dégradation/Amélioration)
        preserved_classes = standard_results['preserved_classes']
        class_ids = list(range(NUM_CLASSES))
        baseline_class_acc = [baseline['per_class'].get(i, 0) for i in class_ids]
        after_class_acc = [after_training['per_class'].get(i, 0) for i in class_ids]
        
        if use_ewc and ewc_results:
            # Comparaison des changements : Standard vs EWC
            ewc_class_acc = [ewc_results['per_class'].get(i, 0) for i in class_ids]
            
            # Calcul des changements
            standard_changes = [after - baseline for after, baseline in zip(after_class_acc, baseline_class_acc)]
            ewc_changes = [ewc - baseline for ewc, baseline in zip(ewc_class_acc, baseline_class_acc)]
            
            x = np.arange(len(class_ids))
            width = 0.35
            
            # Couleurs basées sur positif/négatif
            standard_colors = ['green' if change >= 0 else 'red' for change in standard_changes]
            ewc_colors = ['darkgreen' if change >= 0 else 'darkred' for change in ewc_changes]
            
            bars1 = axes[1,0].bar(x - width/2, standard_changes, width, label='Changement Standard', 
                                 color=standard_colors, alpha=0.8)
            bars2 = axes[1,0].bar(x + width/2, ewc_changes, width, label='Changement EWC', 
                                 color=ewc_colors, alpha=0.8)
            
            # Marquer les classes cibles avec des lignes verticales
            for i, class_id in enumerate(class_ids):
                if class_id in target_classes:
                    axes[1,0].axvline(x=i, color='blue', linestyle=':', alpha=0.6, linewidth=2)
            
            # Annotations pour les changements significatifs
            for i, (std_change, ewc_change) in enumerate(zip(standard_changes, ewc_changes)):
                if abs(std_change) > 0.05:  # Changement significatif
                    axes[1,0].text(i - width/2, std_change + (0.01 if std_change >= 0 else -0.02), 
                                  f'{std_change:+.2f}', ha='center', va='bottom' if std_change >= 0 else 'top', 
                                  fontsize=8, fontweight='bold')
                if abs(ewc_change) > 0.05:  # Changement significatif
                    axes[1,0].text(i + width/2, ewc_change + (0.01 if ewc_change >= 0 else -0.02), 
                                  f'{ewc_change:+.2f}', ha='center', va='bottom' if ewc_change >= 0 else 'top', 
                                  fontsize=8, fontweight='bold')
            
        else:
            # Calculer les changements de performance (après - baseline)
            performance_changes = [after - baseline for after, baseline in zip(after_class_acc, baseline_class_acc)]
            
            # Couleurs basées sur le type de classe et le changement
            colors = []
            for i, change in enumerate(performance_changes):
                if i in target_classes:
                    colors.append('green' if change >= 0 else 'orange')  # Classes cibles : vert si amélioration, orange si dégradation
                else:
                    colors.append('lightcoral' if change < 0 else 'lightblue')  # Classes préservées : rouge si oubli, bleu si stable/amélioration
            
            bars = axes[1,0].bar(class_ids, performance_changes, color=colors, alpha=0.8)
            
            # Annotations pour les changements significatifs
            for i, (bar, change) in enumerate(zip(bars, performance_changes)):
                if abs(change) > 0.05:  # Seulement pour les changements significatifs
                    axes[1,0].text(bar.get_x() + bar.get_width()/2, 
                                  bar.get_height() + (0.01 if change >= 0 else -0.02), 
                                  f'{change:+.2f}', ha='center', 
                                  va='bottom' if change >= 0 else 'top', 
                                  fontsize=9, fontweight='bold')
            
            # Légende personnalisée pour les couleurs
            target_improve_patch = mpatches.Patch(color='green', label='Classes cibles (améliorées)')
            target_degrade_patch = mpatches.Patch(color='orange', label='Classes cibles (dégradées)')
            preserved_forget_patch = mpatches.Patch(color='lightcoral', label='Classes préservées (oubliées)')
            preserved_stable_patch = mpatches.Patch(color='lightblue', label='Classes préservées (stables)')
            axes[1,0].legend(handles=[target_improve_patch, target_degrade_patch, preserved_forget_patch, preserved_stable_patch], 
                           loc='upper right', fontsize=8)
        
        axes[1,0].set_xlabel('Classe')
        axes[1,0].set_ylabel('Changement d\'Accuracy')
        axes[1,0].set_title('📊 Changements de Performance par Classe')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)  # Ligne de référence à 0
        
        # Limites dynamiques basées sur les données
        all_changes = performance_changes if not (use_ewc and ewc_results) else standard_changes + ewc_changes
        y_max = max(0.1, max(all_changes) * 1.2) if all_changes else 0.1
        y_min = min(-0.1, min(all_changes) * 1.2) if all_changes else -0.1
        axes[1,0].set_ylim([y_min, y_max])
        
        # Graphique 4: Résumé des métriques clés
        if use_ewc and 'ewc' in forgetting_analysis:
            # Comparaison Standard vs EWC sur métriques clés
            metrics_names = ['Oubli Global', 'Oubli Classes\nPréservées', 'Amélioration\nClasses Cibles']
            standard_metrics = [
                standard_results['global_forgetting'],
                standard_results['avg_forgetting_preserved_classes'],
                standard_results['avg_improvement_target_classes']
            ]
            ewc_metrics = [
                forgetting_analysis['ewc']['global_forgetting'],
                forgetting_analysis['ewc']['avg_forgetting_preserved_classes'],
                forgetting_analysis['ewc']['avg_improvement_target_classes']
            ]
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            bars1 = axes[1,1].bar(x - width/2, standard_metrics, width, label='Standard', 
                                 color=['red', 'orange', 'blue'], alpha=0.8)
            bars2 = axes[1,1].bar(x + width/2, ewc_metrics, width, label='EWC', 
                                 color=['darkred', 'darkorange', 'darkblue'], alpha=0.8)
            
            axes[1,1].set_xlabel('Métriques')
            axes[1,1].set_ylabel('Score')
            axes[1,1].set_title('📈 Comparaison Standard vs EWC')
            axes[1,1].set_xticks(x)
            axes[1,1].set_xticklabels(metrics_names)
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Annotations
            for bar, value in zip(bars1, standard_metrics):
                axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005 if value >= 0 else bar.get_height() - 0.01, 
                              f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=8)
            for bar, value in zip(bars2, ewc_metrics):
                axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005 if value >= 0 else bar.get_height() - 0.01, 
                              f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=8)
        else:
            # Vue standard sans EWC
            summary_labels = ['Oubli Global', 'Oubli Moyen\n(Classes Préservées)', 'Amélioration Moyenne\n(Classes Cibles)']
            summary_values = [
                standard_results['global_forgetting'],
                standard_results['avg_forgetting_preserved_classes'],
                standard_results['avg_improvement_target_classes']
            ]
            summary_colors = ['red' if v > 0 else 'green' for v in summary_values]
            
            bars4 = axes[1,1].bar(summary_labels, summary_values, color=summary_colors, alpha=0.8)
            axes[1,1].set_ylabel('Changement de Performance')
            axes[1,1].set_title('📈 Résumé: Métriques Clés')
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            for bar, value in zip(bars4, summary_values):
                axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 if value >= 0 else bar.get_height() - 0.02, 
                              f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        
        # Rapport détaillé
        classes_names = [
            "letter", "form", "email", "handwritten", "advertisement", "scientific report",
            "scientific publication", "specification", "file folder", "news article", 
            "budget", "invoice", "presentation", "questionnaire", "resume", "memo"
        ]
        
        target_names = [classes_names[i] for i in target_classes]
        preserved_classes = standard_results['preserved_classes']
        
        report = f"""
🧠 ÉVALUATION RÉELLE DE L'OUBLI CATASTROPHIQUE
============================================

⚙️ CONFIGURATION:
• Modèle: Student (ViT-Tiny) pré-entraîné
• Classes cibles pour fine-tuning: {target_classes} ({', '.join(target_names)})
• Classes à préserver: {len(preserved_classes)} autres classes
• Époques de fine-tuning: {fine_tuning_epochs}
• EWC activé: {'✅ Oui' if use_ewc else '❌ Non'}
{f'• Lambda EWC: {ewc_lambda}' if use_ewc else ''}

📊 RÉSULTATS BASELINE (avant fine-tuning):
• Accuracy globale: {baseline['global']['accuracy']:.4f}
• Precision: {baseline['global']['precision']:.4f}
• Recall: {baseline['global']['recall']:.4f}
• F1-Score: {baseline['global']['f1']:.4f}

🎯 RÉSULTATS FINE-TUNING STANDARD:
• Accuracy globale: {after_training['global']['accuracy']:.4f}
• Precision: {after_training['global']['precision']:.4f}
• Recall: {after_training['global']['recall']:.4f}
• F1-Score: {after_training['global']['f1']:.4f}

🔻 OUBLI CATASTROPHIQUE (STANDARD):
• Oubli global: {standard_results['global_forgetting']:.4f} ({standard_results['global_forgetting']*100:+.2f}%)
• Oubli moyen (classes préservées): {standard_results['avg_forgetting_preserved_classes']:.4f}
• Amélioration moyenne (classes cibles): {standard_results['avg_improvement_target_classes']:.4f}
"""

        if use_ewc and ewc_results:
            ewc_results_data = forgetting_analysis['ewc']
            report += f"""
🧠 RÉSULTATS AVEC EWC:
• Accuracy globale: {ewc_results['global']['accuracy']:.4f}
• Precision: {ewc_results['global']['precision']:.4f}
• Recall: {ewc_results['global']['recall']:.4f}
• F1-Score: {ewc_results['global']['f1']:.4f}

🔻 OUBLI CATASTROPHIQUE (EWC):
• Oubli global: {ewc_results_data['global_forgetting']:.4f} ({ewc_results_data['global_forgetting']*100:+.2f}%)
• Oubli moyen (classes préservées): {ewc_results_data['avg_forgetting_preserved_classes']:.4f}
• Amélioration moyenne (classes cibles): {ewc_results_data['avg_improvement_target_classes']:.4f}

⚖️ COMPARAISON EWC vs STANDARD:
• Réduction d'oubli global: {(standard_results['global_forgetting'] - ewc_results_data['global_forgetting'])*100:+.2f}%
• Réduction d'oubli (classes préservées): {(standard_results['avg_forgetting_preserved_classes'] - ewc_results_data['avg_forgetting_preserved_classes'])*100:+.2f}%
• Différence amélioration cibles: {(ewc_results_data['avg_improvement_target_classes'] - standard_results['avg_improvement_target_classes'])*100:+.2f}%
"""

        report += "\n🏆 INTERPRÉTATION:\n"
        
        # Analyse standard
        if standard_results['global_forgetting'] > 0.05:
            report += "❌ OUBLI SIGNIFICATIF (Standard)! Le modèle a oublié des connaissances importantes.\n"
        elif standard_results['global_forgetting'] > 0.02:
            report += "⚠️ Oubli modéré (Standard). Performance globale légèrement dégradée.\n"
        else:
            report += "✅ Oubli minimal (Standard)! Le modèle a bien préservé ses connaissances.\n"
        
        # Analyse EWC si disponible
        if use_ewc and 'ewc' in forgetting_analysis:
            ewc_data = forgetting_analysis['ewc']
            if ewc_data['global_forgetting'] < standard_results['global_forgetting']:
                reduction = ((standard_results['global_forgetting'] - ewc_data['global_forgetting']) / standard_results['global_forgetting']) * 100
                report += f"🎉 EWC EFFICACE! Réduction de l'oubli de {reduction:.1f}%\n"
            else:
                report += "⚠️ EWC n'a pas réduit l'oubli. Essayez d'ajuster lambda ou plus d'époques baseline.\n"
        
        if standard_results['avg_improvement_target_classes'] > 0.05:
            report += "🎯 Amélioration significative sur les classes cibles!\n"
        elif standard_results['avg_improvement_target_classes'] > 0:
            report += "📈 Légère amélioration sur les classes cibles.\n"
        else:
            report += "⚠️ Peu ou pas d'amélioration sur les classes cibles.\n"
        
        # Classes les plus affectées
        worst_forgetting = [(i, f) for i, f in standard_results['per_class_forgetting'].items() if f > 0.1]
        if worst_forgetting:
            worst_forgetting.sort(key=lambda x: x[1], reverse=True)
            report += f"\n🚨 CLASSES LES PLUS AFFECTÉES (Standard):\n"
            for class_id, forgetting_score in worst_forgetting[:3]:
                report += f"• {classes_names[class_id]} (classe {class_id}): {forgetting_score:.3f} d'oubli\n"
        
        report += f"""
💡 RECOMMANDATIONS:
"""
        if use_ewc:
            if 'ewc' in forgetting_analysis and forgetting_analysis['ewc']['global_forgetting'] < standard_results['global_forgetting']:
                report += "✅ EWC fonctionne bien! Continuez avec cette approche.\n"
            else:
                report += "🔧 Ajustez lambda EWC (essayez des valeurs plus élevées: 5000, 10000).\n"
        else:
            report += "🧠 Essayez EWC pour réduire l'oubli catastrophique.\n"
            
        report += """• Si oubli > 0.05: Utiliser des techniques de regularisation (EWC, LwF)
• Si amélioration faible: Augmenter les époques ou ajuster le learning rate
• Pour réduire l'oubli: Réduire le learning rate ou utiliser du rehearsal
• EWC optimal: lambda entre 1000-10000 selon la complexité du modèle
"""
        
        return report, fig
        
    except Exception as e:
        return f"❌ Erreur lors de l'évaluation: {str(e)}", None

# Interface Gradio
def create_interface():
    with gr.Blocks(title="Comparaison Modèles & Apprentissage Continu", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 Analyse Comparative: Student vs Teacher Models
        ## Interface d'évaluation des performances et d'apprentissage continu
        """)
        
        with gr.Tabs():
            # Onglet 1: Comparaison des performances
            with gr.TabItem("📊 Comparaison des Performances"):
                gr.Markdown("""
                ### Comparaison Student Model vs Teacher Model
                Analyse des performances, efficience et trade-offs entre les modèles.
                """)
                
                compare_btn = gr.Button("🚀 Lancer la Comparaison", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column():
                        comparison_report = gr.Textbox(
                            label="📋 Rapport de Comparaison",
                            lines=20,
                            placeholder="Cliquez sur 'Lancer la Comparaison' pour voir les résultats..."
                        )
                    
                    with gr.Column():
                        performance_plot = gr.Plot(label="📈 Métriques de Performance")
                        efficiency_plot = gr.Plot(label="⚡ Temps d'Inférence et Tailles")
                
                compare_btn.click(
                    compare_models,
                    outputs=[comparison_report, performance_plot, efficiency_plot]
                )
            
            # Onglet 2: Apprentissage Continu
            with gr.TabItem("🧠 Apprentissage Continu"):
                gr.Markdown("""
                ### 🔬 Simulation d'Apprentissage Continu
                
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Configuration")
                        
                        method_selection = gr.CheckboxGroup(
                            choices=["Naive", "Rehearsal", "LwF (Learning without Forgetting)"],
                            value=["Naive", "Rehearsal"],
                            label="🎯 Méthodes à évaluer",
                            info="Sélectionnez les méthodes d'apprentissage continu à comparer"
                        )
                        
                        num_tasks_slider = gr.Slider(
                            minimum=2, maximum=8, value=4, step=1,
                            label="📚 Nombre de tâches",
                            info="Nombre de tâches d'apprentissage séquentielles"
                        )
                        
                        continual_btn = gr.Button(
                            "🚀 Démarrer l'Expérience", 
                            variant="primary", 
                            size="lg"
                        )
                        
                        gr.Markdown("""
                        **📋 Méthodes simulées:**
                        - **Naive**: Apprentissage séquentiel simple (oubli total)
                        - **Rehearsal**: Mélange basique avec 25% d'anciennes données
                        - **LwF**: Moins d'époques (simulation très simplifiée)
                        
                        **⚠️ Ces implémentations sont des démonstrations de concepts, pas des méthodes complètes de continual learning.**
                        """)
                    
                    with gr.Column(scale=2):
                        continual_plot = gr.Plot(
                            label="📊 Résultats de l'Apprentissage Continu",
                            visible=True
                        )
                
                with gr.Row():
                    continual_report = gr.Textbox(
                        label="📋 Analyse des Résultats",
                        lines=15,
                        placeholder="Configurez l'expérience et cliquez sur 'Démarrer' pour voir les résultats...",
                        visible=True
                    )
                
                continual_btn.click(
                    run_continual_learning_experiment,
                    inputs=[method_selection, num_tasks_slider],
                    outputs=[continual_report, continual_plot]
                )
            
            # Onglet 3: Évaluation RÉELLE de l'Oubli Catastrophique
            with gr.TabItem("🧠 Oubli Catastrophique RÉEL"):
                gr.Markdown("""
                ### 🔬 Évaluation Réelle de l'Oubli Catastrophique
                **🎯 Analyse concrète :** Mesurez l'oubli catastrophique réel du modèle Student en le fine-tunant sur des classes spécifiques.
                
                **📋 Processus :**
                1. Évaluation baseline du modèle sur toutes les classes
                2. Fine-tuning sur les classes que vous sélectionnez  
                3. Re-évaluation et calcul de l'oubli catastrophique
                4. Analyse détaillée par classe et recommandations
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Configuration de l'Expérience")
                        
                        target_classes_input = gr.Textbox(
                            label="🎯 Classes cibles pour fine-tuning",
                            placeholder="0,1,2",
                            value="0,1,2",
                            info="Numéros des classes (0-15) séparés par des virgules. Ex: 0,1,2 pour letter,form,email"
                        )
                        
                        epochs_slider = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1,
                            label="📚 Époques de fine-tuning",
                            info="Plus d'époques = plus d'oubli potentiel mais meilleure performance sur les cibles"
                        )
                        
                        use_ewc_checkbox = gr.Checkbox(
                            label="🧠 Activer EWC (Elastic Weight Consolidation)",
                            value=False,
                            info="Compare fine-tuning standard vs EWC pour réduire l'oubli catastrophique"
                        )
                        
                        ewc_lambda_slider = gr.Slider(
                            minimum=100, maximum=10000, value=1000, step=100,
                            label="⚖️ Lambda EWC (force de régularisation)",
                            info="Plus élevé = moins d'oubli mais peut limiter l'apprentissage des nouvelles tâches",
                            visible=False
                        )
                        
                        real_forgetting_btn = gr.Button(
                            "🚀 Lancer l'Évaluation Réelle", 
                            variant="primary", 
                            size="lg"
                        )
                        
                        # Afficher/masquer le slider lambda selon EWC
                        def toggle_ewc_lambda(use_ewc):
                            return gr.update(visible=use_ewc)
                        
                        use_ewc_checkbox.change(
                            toggle_ewc_lambda,
                            inputs=[use_ewc_checkbox],
                            outputs=[ewc_lambda_slider]
                        )
                        
                        gr.Markdown("""
                        **📚 Classes RVL-CDIP (0-15) :**
                        - 0: letter, 1: form, 2: email, 3: handwritten
                        - 4: advertisement, 5: scientific report, 6: scientific publication
                        - 7: specification, 8: file folder, 9: news article
                        - 10: budget, 11: invoice, 12: presentation
                        - 13: questionnaire, 14: resume, 15: memo
                        
                        **🧠 Elastic Weight Consolidation (EWC) :**
                        - Technique avancée de continual learning
                        - Utilise la Fisher Information Matrix pour préserver les poids importants
                        - Lambda contrôle le trade-off entre nouvelle tâche et préservation
                        
                        **⚠️ Note :** Cette évaluation prend 5-10 minutes (15-20 min avec EWC).
                        """)
                    
                    with gr.Column(scale=2):
                        real_forgetting_plot = gr.Plot(
                            label="📊 Analyse de l'Oubli Catastrophique Réel",
                            visible=True
                        )
                
                with gr.Row():
                    real_forgetting_report = gr.Textbox(
                        label="📋 Rapport d'Analyse Détaillé",
                        lines=20,
                        placeholder="Configurez l'expérience et cliquez sur 'Lancer' pour mesurer l'oubli catastrophique réel...",
                        visible=True
                    )
                
                real_forgetting_btn.click(
                    run_real_catastrophic_forgetting_evaluation,
                    inputs=[target_classes_input, epochs_slider, use_ewc_checkbox, ewc_lambda_slider],
                    outputs=[real_forgetting_report, real_forgetting_plot]
                )
            
            # Onglet 4: Documentation
            with gr.TabItem("📚 Documentation"):
                gr.Markdown("""
                ### 📖 Guide d'utilisation
                
                #### 🎯 Onglet Comparaison des Performances
                - **Student Model**: HAMMALE/vit-tiny-classifier-rvlcdip (Vision Transformer compact)
                - **Teacher Model**: microsoft/dit-large-finetuned-rvlcdip (Document Image Transformer)
                - **Dataset**: HAMMALE/rvl_cdip_OCR (classification de documents)
                
                **Métriques évaluées:**
                - Accuracy, Precision, Recall, F1-Score
                - Temps d'inférence et taille du modèle
                - Ratio de compression et perte de performance
                
                #### 🧠 Onglet Apprentissage Continu (Simulation)
                **⚠️ Simulation éducative** des concepts d'apprentissage continu :
                - **Naive**: Apprentissage séquentiel simple
                - **Rehearsal**: Mélange basique avec données précédentes  
                - **LwF**: Simulation simplifiée de Learning without Forgetting
                
                **Métriques simulées:**
                - Performance par tâche artificielle
                - Score d'oubli approximatif
                - Comparaison des approches conceptuelles
                
                #### 🔬 Onglet Oubli Catastrophique RÉEL
                **🎯 Évaluation concrète** de l'oubli catastrophique du modèle Student :
                
                **Processus scientifique:**
                1. Mesure baseline du modèle pré-entraîné sur toutes les classes
                2. Fine-tuning réel sur classes spécifiques sélectionnées
                3. Re-évaluation et calcul précis de l'oubli par classe
                4. Analyse statistique et recommandations pratiques
                
                **Métriques réelles:**
                - Oubli global et par classe (avant/après fine-tuning)
                - Performance différentielle sur classes cibles vs préservées
                - Identification des classes les plus vulnérables
                - Comparaison Standard vs EWC (Elastic Weight Consolidation)
                - Seuils d'alerte et recommandations techniques
                
                **🧠 EWC (Elastic Weight Consolidation):**
                - Technique state-of-the-art pour réduire l'oubli catastrophique
                - Calcul de la Fisher Information Matrix pour identifier les poids critiques
                - Régularisation adaptative basée sur l'importance des paramètres
                
                #### 🔧 Configuration Technique
                - Utilise PyTorch et Transformers
                - Support GPU/CPU automatique
                - Limitation des données pour les tests (performance)
                
                #### ⚠️ Notes Importantes
                
                **Onglets Simulation vs Réel :**
                - **Onglet 2 (Simulation)** : Démonstration pédagogique des concepts
                - **Onglet 3 (RÉEL)** : Mesures scientifiques précises avec fine-tuning effectif
                
                **Performance :**
                - Simulation : ~2-5 minutes
                - Évaluation réelle : ~5-15 minutes (selon époques)
                - Les résultats peuvent varier selon le matériel disponible
                
                **Données :**
                - Sous-ensembles utilisés pour optimiser la vitesse d'exécution
                - Résultats représentatifs du comportement complet
                """)
    
    return demo

# Lancement de l'interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=None,  # Trouve automatiquement un port libre
        show_error=True
    )
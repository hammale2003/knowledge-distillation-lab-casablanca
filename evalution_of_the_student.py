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




# Évaluation RÉELLE de l'oubli catastrophique
class CatastrophicForgettingEvaluator:
    def __init__(self, model, processor):
        self.original_model = copy.deepcopy(model)
        self.model = model
        self.processor = processor
        self.baseline_performance = {}
        self.after_training_performance = {}
        self.continual_performance = {}
        self.fisher_information = {}
        self.optimal_params = {}
        self.teacher_logits = {}  # Pour LwF
        self.continual_method = None
        
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
        """Fine-tune le modèle sur un sous-ensemble de classes (méthode standard)"""
        print(f"🎯 Fine-tuning STANDARD sur les classes: {target_classes}")
        
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
        """Évaluation après fine-tuning standard"""
        print("📈 Évaluation après fine-tuning standard...")
        
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
    
    def prepare_continual_learning(self, dataset, method='ewc', sample_size=200):
        """Préparer les données nécessaires pour l'apprentissage continu"""
        print(f"🧠 Préparation pour l'apprentissage continu: {method.upper()}")
        
        self.continual_method = method
        
        if method == 'ewc':
            self.compute_fisher_information(dataset, sample_size)
        elif method == 'lwf':
            self.compute_teacher_logits(dataset, sample_size)
        elif method == 'mas':
            self.compute_mas_importance(dataset, sample_size)
        
    def compute_fisher_information(self, dataset, sample_size=200):
        """Calcule la Fisher Information Matrix pour EWC (version améliorée)"""
        print("🧮 Calcul de la Fisher Information Matrix (EWC)...")
        
        # Prendre un échantillon équilibré pour le calcul
        sample_data = dataset['train'].select(range(min(sample_size, len(dataset['train']))))
        dataloader = DataLoader(sample_data, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
        
        self.model.eval()
        self.model.to(DEVICE)
        
        # Sauvegarder les paramètres optimaux
        self.optimal_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone().detach()
        
        # Initialiser Fisher Information
        self.fisher_information = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_information[name] = torch.zeros_like(param.data).to(DEVICE)
        
        criterion = nn.CrossEntropyLoss()
        total_samples = 0
        
        # Calculer Fisher Information avec plusieurs passes
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 50:
                break
                
            images = batch['image']
            labels = torch.tensor(batch['label'], dtype=torch.long).to(DEVICE)
            images = ensure_rgb_images(images)
            
            inputs = self.processor(images, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # Calculer gradients pour chaque échantillon
            for i in range(len(labels)):
                self.model.zero_grad()
                single_input = {k: v[i:i+1] for k, v in inputs.items()}
                single_label = labels[i:i+1]
                
                outputs = self.model(**single_input)
                loss = criterion(outputs.logits, single_label)
                loss.backward()
                
                # Accumuler Fisher Information (gradient^2)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.fisher_information[name] += param.grad.data ** 2
                
                total_samples += 1
        
        # Normaliser par le nombre d'échantillons
        for name in self.fisher_information:
            self.fisher_information[name] /= total_samples
            # Ajouter une petite constante pour éviter les zéros
            self.fisher_information[name] += 1e-8
        
        print(f"✅ Fisher Information calculée sur {total_samples} échantillons")
        
    def compute_teacher_logits(self, dataset, sample_size=200):
        """Calcule les logits du modèle baseline pour LwF"""
        print("🎓 Calcul des logits du modèle teacher (LwF)...")
        
        # Utiliser le modèle baseline comme teacher
        teacher_model = copy.deepcopy(self.original_model)
        teacher_model.eval()
        teacher_model.to(DEVICE)
        
        # Prendre un échantillon pour calculer les logits
        sample_data = dataset['train'].select(range(min(sample_size, len(dataset['train']))))
        dataloader = DataLoader(sample_data, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
        
        self.teacher_logits = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                images = batch['image']
                labels = batch['label']
                images = ensure_rgb_images(images)
                
                inputs = self.processor(images, return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                outputs = teacher_model(**inputs)
                
                # Stocker les logits pour chaque échantillon
                for i, label in enumerate(labels):
                    sample_idx = batch_idx * 8 + i
                    self.teacher_logits[sample_idx] = outputs.logits[i].cpu().detach()
        
        print(f"✅ Logits teacher calculés pour {len(self.teacher_logits)} échantillons")
        
    def compute_mas_importance(self, dataset, sample_size=200):
        """Calcule l'importance des paramètres pour MAS (Memory Aware Synapses)"""
        print("🧠 Calcul de l'importance MAS...")
        
        # Prendre un échantillon
        sample_data = dataset['train'].select(range(min(sample_size, len(dataset['train']))))
        dataloader = DataLoader(sample_data, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
        
        self.model.eval()
        self.model.to(DEVICE)
        
        # Sauvegarder les paramètres optimaux
        self.optimal_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone().detach()
        
        # Initialiser l'importance MAS
        self.mas_importance = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.mas_importance[name] = torch.zeros_like(param.data).to(DEVICE)
        
        total_samples = 0
        
        # Calculer l'importance basée sur les gradients de l'output
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 50:
                break
                
            images = batch['image']
            labels = torch.tensor(batch['label'], dtype=torch.long).to(DEVICE)
            images = ensure_rgb_images(images)
            
            inputs = self.processor(images, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # Calculer gradients pour chaque échantillon
            for i in range(len(labels)):
                self.model.zero_grad()
                single_input = {k: v[i:i+1] for k, v in inputs.items()}
                
                outputs = self.model(**single_input)
                # Utiliser la norme L2 de l'output au lieu de la loss
                l2_norm = torch.norm(outputs.logits, p=2)
                l2_norm.backward()
                
                # Accumuler l'importance (gradient absolu)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.mas_importance[name] += param.grad.data.abs()
                
                total_samples += 1
        
        # Normaliser
        for name in self.mas_importance:
            self.mas_importance[name] /= total_samples
            self.mas_importance[name] += 1e-8
        
        print(f"✅ Importance MAS calculée sur {total_samples} échantillons")
    
    def fine_tune_with_continual_learning(self, dataset, target_classes, epochs=3, 
                                        method='ewc', reg_lambda=1000):
        """Fine-tune avec techniques d'apprentissage continu"""
        
        # Définir la lambda selon la méthode
        if method == 'ewc':
            effective_lambda = reg_lambda
        elif method == 'lwf':
            effective_lambda = 0.5  # Lambda fixe pour LwF
        elif method == 'mas':
            effective_lambda = 1.0  # Lambda fixe pour MAS
        else:
            effective_lambda = 0.0
            
        print(f"🧠 Fine-tuning avec {method.upper()} (λ={effective_lambda}) sur les classes: {target_classes}")
        
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
            epoch_reg_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                images = batch['image']
                labels = torch.tensor(batch['label'], dtype=torch.long).to(DEVICE)
                images = ensure_rgb_images(images)
                
                inputs = self.processor(images, return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                
                # Loss standard
                standard_loss = base_criterion(outputs.logits, labels)
                
                # Loss de régularisation selon la méthode
                if method == 'ewc':
                    reg_loss = self._compute_ewc_loss()
                elif method == 'lwf':
                    reg_loss = self._compute_lwf_loss(outputs.logits, batch_idx)
                elif method == 'mas':
                    reg_loss = self._compute_mas_loss()
                else:
                    reg_loss = torch.tensor(0.0, device=DEVICE)
                
                # Loss totale
                total_loss = standard_loss + effective_lambda * reg_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += standard_loss.item()
                epoch_reg_loss += reg_loss.item()
                num_batches += 1
            
            avg_standard_loss = epoch_loss / num_batches
            avg_reg_loss = epoch_reg_loss / num_batches
            print(f"  Époque {epoch+1}/{epochs}, Loss standard: {avg_standard_loss:.4f}, "
                  f"Loss {method.upper()}: {avg_reg_loss:.6f}")
    
    def _compute_ewc_loss(self):
        """Calcule la perte EWC"""
        ewc_loss = torch.tensor(0.0, device=DEVICE)
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_information and name in self.optimal_params and param.requires_grad:
                fisher = self.fisher_information[name].to(DEVICE)
                optimal = self.optimal_params[name].to(DEVICE)
                ewc_loss += (fisher * (param - optimal) ** 2).sum()
        
        return ewc_loss
    
    def _compute_lwf_loss(self, student_logits, batch_idx):
        """Calcule la perte LwF (Learning without Forgetting)"""
        lwf_loss = torch.tensor(0.0, device=DEVICE)
        
        # Utiliser les logits du teacher stockés pour la distillation
        if hasattr(self, 'teacher_logits') and self.teacher_logits:
            temperature = 4.0
            kl_loss = nn.KLDivLoss(reduction='batchmean')
            
            # Utiliser les logits du teacher stockés
            batch_size = student_logits.size(0)
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            # Vérifier si on a assez de teacher logits
            if end_idx <= len(self.teacher_logits):
                teacher_batch_logits = self.teacher_logits[start_idx:end_idx].to(DEVICE)
                
                # Distillation avec température
                student_soft = F.log_softmax(student_logits / temperature, dim=1)
                teacher_soft = F.softmax(teacher_batch_logits / temperature, dim=1)
                
                lwf_loss = kl_loss(student_soft, teacher_soft) * (temperature ** 2)
        
        return lwf_loss
    
    def _compute_mas_loss(self):
        """Calcule la perte MAS"""
        mas_loss = torch.tensor(0.0, device=DEVICE)
        
        for name, param in self.model.named_parameters():
            if name in self.mas_importance and name in self.optimal_params and param.requires_grad:
                importance = self.mas_importance[name].to(DEVICE)
                optimal = self.optimal_params[name].to(DEVICE)
                mas_loss += (importance * (param - optimal) ** 2).sum()
        
        return mas_loss
    
    def evaluate_continual_performance(self, dataset):
        """Évaluation après fine-tuning avec apprentissage continu"""
        print(f"📈 Évaluation après fine-tuning avec {self.continual_method.upper()}...")
        
        test_data = dataset['test'].select(range(min(1000, len(dataset['test']))))
        dataloader = DataLoader(test_data, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
        
        # Performance globale
        accuracy, precision, recall, f1, _ = evaluate_model(self.model, self.processor, dataloader, DEVICE)
        
        # Performance par classe
        class_performance = self.evaluate_per_class(dataloader)
        
        self.continual_performance = {
            'global': {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1},
            'per_class': class_performance
        }
        
        return self.continual_performance
    
    def calculate_forgetting(self, target_classes, include_continual=False):
        """Calcul de l'oubli catastrophique"""
        if not self.baseline_performance or not self.after_training_performance:
            return None
            
        # Résultats standards
        results = {
            'standard': self._calculate_forgetting_for_method(
                self.baseline_performance, self.after_training_performance, target_classes, 'Standard'
            )
        }
        
        # Résultats avec apprentissage continu si disponibles
        if include_continual and self.continual_performance:
            method_name = f"{self.continual_method.upper()}" if self.continual_method else "Continual"
            results['continual'] = self._calculate_forgetting_for_method(
                self.baseline_performance, self.continual_performance, target_classes, method_name
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

def run_real_catastrophic_forgetting_evaluation(target_classes_str, fine_tuning_epochs, continual_method, reg_lambda):
    """Évaluation de l'oubli catastrophique"""
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
        
        # 4. Fine-tuning avec apprentissage continu si demandé
        continual_results = None
        if continual_method != 'none':
            print(f"\n🧠 === FINE-TUNING AVEC {continual_method.upper()} ===")
            # Restaurer le modèle original
            evaluator.model = copy.deepcopy(original_model)
            
            # Préparer les données nécessaires pour l'apprentissage continu
            evaluator.prepare_continual_learning(dataset, method=continual_method)
            
            # Fine-tuning avec la méthode d'apprentissage continu
            evaluator.fine_tune_with_continual_learning(
                dataset, target_classes, epochs=fine_tuning_epochs, 
                method=continual_method, reg_lambda=reg_lambda
            )
            continual_results = evaluator.evaluate_continual_performance(dataset)
        
        # 5. Calcul de l'oubli catastrophique
        forgetting_analysis = evaluator.calculate_forgetting(target_classes, include_continual=(continual_method != 'none'))
        
        # Création des graphiques
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        title = '🧠 Analyse de l\'Oubli Catastrophique'
        if continual_method != 'none':
            title += f' - Comparaison Standard vs {continual_method.upper()}'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Préparation des données
        standard_results = forgetting_analysis['standard']
        
        # Graphique 1: Comparaison performance globale
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        baseline_values = [baseline['global'][m] for m in metrics]
        after_values = [after_training['global'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        
        if continual_method != 'none' and continual_results:
            # Comparaison à 3 barres : Baseline, Standard, Méthode Continue
            continual_values = [continual_results['global'][m] for m in metrics]
            width = 0.25
            
            bars1 = axes[0,0].bar(x - width, baseline_values, width, label='Baseline', color='skyblue', alpha=0.8)
            bars2 = axes[0,0].bar(x, after_values, width, label='Fine-tuning Standard', color='lightcoral', alpha=0.8)
            bars3 = axes[0,0].bar(x + width, continual_values, width, label=f'Fine-tuning {continual_method.upper()}', color='lightgreen', alpha=0.8)
            
            # Annotations pour 3 barres
            for i, (bar1, bar2, bar3, base_val, std_val, cont_val) in enumerate(zip(bars1, bars2, bars3, baseline_values, after_values, continual_values)):
                axes[0,0].text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01, 
                              f'{base_val:.3f}', ha='center', va='bottom', fontsize=8)
                axes[0,0].text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01, 
                              f'{std_val:.3f}', ha='center', va='bottom', fontsize=8)
                axes[0,0].text(bar3.get_x() + bar3.get_width()/2, bar3.get_height() + 0.01, 
                              f'{cont_val:.3f}', ha='center', va='bottom', fontsize=8)
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
        if continual_method != 'none' and 'continual' in forgetting_analysis:
            # Comparaison Standard vs Méthode Continue
            methods = ['Standard', continual_method.upper()]
            global_forgetting_values = [
                forgetting_analysis['standard']['global_forgetting'],
                forgetting_analysis['continual']['global_forgetting']
            ]
            avg_forgetting_values = [
                forgetting_analysis['standard']['avg_forgetting_preserved_classes'],
                forgetting_analysis['continual']['avg_forgetting_preserved_classes']
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
        
        if continual_method != 'none' and continual_results:
            # Comparaison des changements : Standard vs Méthode Continue
            continual_class_acc = [continual_results['per_class'].get(i, 0) for i in class_ids]
            
            # Calcul des changements
            standard_changes = [after - baseline for after, baseline in zip(after_class_acc, baseline_class_acc)]
            continual_changes = [continual - baseline for continual, baseline in zip(continual_class_acc, baseline_class_acc)]
            
            x = np.arange(len(class_ids))
            width = 0.35
            
            # Couleurs basées sur positif/négatif
            standard_colors = ['green' if change >= 0 else 'red' for change in standard_changes]
            continual_colors = ['darkgreen' if change >= 0 else 'darkred' for change in continual_changes]
            
            bars1 = axes[1,0].bar(x - width/2, standard_changes, width, label='Changement Standard', 
                                 color=standard_colors, alpha=0.8)
            bars2 = axes[1,0].bar(x + width/2, continual_changes, width, label=f'Changement {continual_method.upper()}', 
                                 color=continual_colors, alpha=0.8)
            
            # Marquer les classes cibles avec des lignes verticales
            for i, class_id in enumerate(class_ids):
                if class_id in target_classes:
                    axes[1,0].axvline(x=i, color='blue', linestyle=':', alpha=0.6, linewidth=2)
            
            # Annotations pour les changements significatifs
            for i, (std_change, cont_change) in enumerate(zip(standard_changes, continual_changes)):
                if abs(std_change) > 0.05:  # Changement significatif
                    axes[1,0].text(i - width/2, std_change + (0.01 if std_change >= 0 else -0.02), 
                                  f'{std_change:+.2f}', ha='center', va='bottom' if std_change >= 0 else 'top', 
                                  fontsize=8, fontweight='bold')
                if abs(cont_change) > 0.05:  # Changement significatif
                    axes[1,0].text(i + width/2, cont_change + (0.01 if cont_change >= 0 else -0.02), 
                                  f'{cont_change:+.2f}', ha='center', va='bottom' if cont_change >= 0 else 'top', 
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
        all_changes = performance_changes if not (continual_method != 'none' and continual_results) else standard_changes + continual_changes
        y_max = max(0.1, max(all_changes) * 1.2) if all_changes else 0.1
        y_min = min(-0.1, min(all_changes) * 1.2) if all_changes else -0.1
        axes[1,0].set_ylim([y_min, y_max])
        
        # Graphique 4: Résumé des métriques clés
        if continual_method != 'none' and 'continual' in forgetting_analysis:
            # Comparaison Standard vs Méthode Continue sur métriques clés
            metrics_names = ['Oubli Global', 'Oubli Classes\nPréservées', 'Amélioration\nClasses Cibles']
            standard_metrics = [
                standard_results['global_forgetting'],
                standard_results['avg_forgetting_preserved_classes'],
                standard_results['avg_improvement_target_classes']
            ]
            continual_metrics = [
                forgetting_analysis['continual']['global_forgetting'],
                forgetting_analysis['continual']['avg_forgetting_preserved_classes'],
                forgetting_analysis['continual']['avg_improvement_target_classes']
            ]
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            bars1 = axes[1,1].bar(x - width/2, standard_metrics, width, label='Standard', 
                                 color=['red', 'orange', 'blue'], alpha=0.8)
            bars2 = axes[1,1].bar(x + width/2, continual_metrics, width, label=continual_method.upper(), 
                                 color=['darkred', 'darkorange', 'darkblue'], alpha=0.8)
            
            axes[1,1].set_xlabel('Métriques')
            axes[1,1].set_ylabel('Score')
            axes[1,1].set_title(f'📈 Comparaison Standard vs {continual_method.upper()}')
            axes[1,1].set_xticks(x)
            axes[1,1].set_xticklabels(metrics_names)
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Annotations
            for bar, value in zip(bars1, standard_metrics):
                axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005 if value >= 0 else bar.get_height() - 0.01, 
                              f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=8)
            for bar, value in zip(bars2, continual_metrics):
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
• Méthode d'apprentissage continu: {continual_method.upper() if continual_method != 'none' else 'Aucune'}
{f'• Lambda régularisation: {reg_lambda}' if continual_method != 'none' else ''}

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

        if continual_method != 'none' and continual_results:
            continual_results_data = forgetting_analysis['continual']
            report += f"""
🧠 RÉSULTATS AVEC {continual_method.upper()}:
• Accuracy globale: {continual_results['global']['accuracy']:.4f}
• Precision: {continual_results['global']['precision']:.4f}
• Recall: {continual_results['global']['recall']:.4f}
• F1-Score: {continual_results['global']['f1']:.4f}

🔻 OUBLI CATASTROPHIQUE ({continual_method.upper()}):
• Oubli global: {continual_results_data['global_forgetting']:.4f} ({continual_results_data['global_forgetting']*100:+.2f}%)
• Oubli moyen (classes préservées): {continual_results_data['avg_forgetting_preserved_classes']:.4f}
• Amélioration moyenne (classes cibles): {continual_results_data['avg_improvement_target_classes']:.4f}

⚖️ COMPARAISON {continual_method.upper()} vs STANDARD:
• Réduction d'oubli global: {(standard_results['global_forgetting'] - continual_results_data['global_forgetting'])*100:+.2f}%
• Réduction d'oubli (classes préservées): {(standard_results['avg_forgetting_preserved_classes'] - continual_results_data['avg_forgetting_preserved_classes'])*100:+.2f}%
• Différence amélioration cibles: {(continual_results_data['avg_improvement_target_classes'] - standard_results['avg_improvement_target_classes'])*100:+.2f}%
"""

        report += "\n🏆 INTERPRÉTATION:\n"
        
        # Analyse standard
        if standard_results['global_forgetting'] > 0.05:
            report += "❌ OUBLI SIGNIFICATIF (Standard)! Le modèle a oublié des connaissances importantes.\n"
        elif standard_results['global_forgetting'] > 0.02:
            report += "⚠️ Oubli modéré (Standard). Performance globale légèrement dégradée.\n"
        else:
            report += "✅ Oubli minimal (Standard)! Le modèle a bien préservé ses connaissances.\n"
        
        # Analyse de la méthode d'apprentissage continu si disponible
        if continual_method != 'none' and 'continual' in forgetting_analysis:
            continual_data = forgetting_analysis['continual']
            if continual_data['global_forgetting'] < standard_results['global_forgetting']:
                reduction = ((standard_results['global_forgetting'] - continual_data['global_forgetting']) / standard_results['global_forgetting']) * 100
                report += f"🎉 {continual_method.upper()} EFFICACE! Réduction de l'oubli de {reduction:.1f}%\n"
            else:
                report += f"⚠️ {continual_method.upper()} n'a pas réduit l'oubli. Essayez d'ajuster lambda ou plus d'époques baseline.\n"
        
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
        if continual_method != 'none':
            if 'continual' in forgetting_analysis and forgetting_analysis['continual']['global_forgetting'] < standard_results['global_forgetting']:
                report += f"✅ {continual_method.upper()} fonctionne bien! Continuez avec cette approche.\n"
            else:
                report += f"🔧 Ajustez lambda {continual_method.upper()} (essayez des valeurs plus élevées: 5000, 10000).\n"
        else:
            report += "🧠 Essayez une méthode d'apprentissage continu pour réduire l'oubli catastrophique.\n"
            
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
            
            # Onglet 2: Évaluation RÉELLE de l'Oubli Catastrophique
            with gr.TabItem("🧠 Oubli Catastrophique"):
                gr.Markdown("""
                ### 🔬 Évaluation de l'Oubli Catastrophique""")
                
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
                        
                        continual_method_dropdown = gr.Dropdown(
                            choices=[
                                ("Aucune méthode (fine-tuning standard seulement)", "none"),
                                ("EWC - Elastic Weight Consolidation", "ewc"),
                                ("LwF - Learning without Forgetting", "lwf"),
                                ("MAS - Memory Aware Synapses", "mas")
                            ],
                            value="none",
                            label="🧠 Méthode d'apprentissage continu",
                            info="Choisir une technique pour réduire l'oubli catastrophique"
                        )
                        
                        reg_lambda_slider = gr.Slider(
                            minimum=100, maximum=10000, value=1000, step=100,
                            label="⚖️ Lambda régularisation (force de régularisation)",
                            info="Plus élevé = moins d'oubli mais peut limiter l'apprentissage des nouvelles tâches",
                            visible=False
                        )
                        
                        real_forgetting_btn = gr.Button(
                            "🚀 Lancer l'Évaluation Réelle", 
                            variant="primary", 
                            size="lg"
                        )
                        
                        # Afficher/masquer le slider lambda selon la méthode sélectionnée
                        def toggle_lambda_slider(continual_method):
                            return gr.update(visible=continual_method == "ewc")
                        
                        continual_method_dropdown.change(
                            toggle_lambda_slider,
                            inputs=[continual_method_dropdown],
                            outputs=[reg_lambda_slider]
                        )
                        
                        gr.Markdown("""
                        **📚 Classes RVL-CDIP (0-15) :**
                        - 0: letter, 1: form, 2: email, 3: handwritten
                        - 4: advertisement, 5: scientific report, 6: scientific publication
                        - 7: specification, 8: file folder, 9: news article
                        - 10: budget, 11: invoice, 12: presentation
                        - 13: questionnaire, 14: resume, 15: memo
                        
                        **🧠 Méthodes d'apprentissage continu :**
                        - **EWC** : Utilise la Fisher Information Matrix pour préserver les poids importants
                        - **LwF** : Distillation de connaissances avec le modèle original comme teacher
                        - **MAS** : Memory Aware Synapses basé sur l'importance des gradients
                        
                        **⚖️ Lambda** : Contrôle le trade-off entre nouvelle tâche et préservation (uniquement pour EWC)
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
                    inputs=[target_classes_input, epochs_slider, continual_method_dropdown, reg_lambda_slider],
                    outputs=[real_forgetting_report, real_forgetting_plot]
                )
            
            # Onglet 3: Documentation
            with gr.TabItem("📚 Documentation"):
                gr.Markdown("""
                ### 📖 Guide d'utilisation
                
                #### 🎯 Onglet 1: Comparaison des Performances
                - **Student Model**: HAMMALE/vit-tiny-classifier-rvlcdip (Vision Transformer compact)
                - **Teacher Model**: microsoft/dit-large-finetuned-rvlcdip (Document Image Transformer)
                - **Dataset**: HAMMALE/rvl_cdip_OCR (classification de documents)
                
                **Métriques évaluées:**
                - Accuracy, Precision, Recall, F1-Score
                - Temps d'inférence et taille du modèle
                - Ratio de compression et perte de performance
                
                #### 🧠 Onglet 2: Oubli Catastrophique RÉEL
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
                - Comparaison Standard vs méthodes d'apprentissage continu
                - Seuils d'alerte et recommandations techniques
                
                **🧠 Méthodes d'apprentissage continu disponibles:**
                - **EWC (Elastic Weight Consolidation)** : Utilise la Fisher Information Matrix pour préserver les poids importants
                - **LwF (Learning without Forgetting)** : Distillation de connaissances avec le modèle original comme teacher
                - **MAS (Memory Aware Synapses)** : Basé sur l'importance des gradients de sortie
                
                **⚖️ Paramètre Lambda** : Contrôle le trade-off entre nouvelle tâche et préservation des anciennes (uniquement pour EWC)
                
                #### 🔧 Configuration Technique
                - Utilise PyTorch et Transformers
                - Support GPU/CPU automatique
                - Limitation des données pour optimiser la vitesse d'exécution
                
                #### ⚠️ Notes Importantes
                
                **Performance :**
                - Comparaison des modèles : ~2-5 minutes
                - Évaluation oubli catastrophique : ~5-15 minutes (selon époques et méthode)
                - Les résultats peuvent varier selon le matériel disponible
                
                **Données :**
                - Sous-ensembles utilisés pour optimiser la vitesse d'exécution
                - Résultats représentatifs du comportement complet
                - Classes RVL-CDIP : 16 classes de documents (letters, forms, emails, etc.)
                
                **Recommandations d'utilisation :**
                - Commencez avec 2-3 classes cibles pour tester
                - Utilisez 3-5 époques pour un bon compromis temps/résultats
                - Testez différentes méthodes continual learning pour comparer
                - Lambda entre 1000-5000 généralement optimal
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

"""
Generalizable para evaluacion de modelos
"""

# Standard library
import os
from datetime import datetime
from pathlib import Path

# Data & Numerical
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    )

from sklearn.preprocessing import label_binarize

import warnings
warnings.filterwarnings('ignore')


class ResultsAnalyzerMulticlase:
    """
    Evaluación de modelos generalizable
    """
    def __init__(self):
        """
        Inicializa el analizador de resultados
        """
        self.resultados = []


    def _evaluar_modelo(self, y_test, y_pred, y_pred_proba=None, nombre_modelo=None, 
                   mostrar_grafico=True, guardar_grafico=False, ruta_guardado=None):
        """
        Evalúa un modelo de clasificación binaria y muestra métricas completas.
        Utiliza ROC AUC #One-vs-Rest (OvR) que calcula una curva ROC para cada clase, comparando esa clase vs. todas las demás.
        Después promedia las áreas bajo las curvas.
        
        
        Parameters:
        -----------
        y_test : array-like
            Valores reales del conjunto de test
        y_pred : array-like
            Predicciones del modelo
        y_pred_proba : array-like, optional
            Probabilidades predichas (para calcular ROC-AUC)
        nombre_modelo : str, default="Modelo"
            Nombre del modelo para mostrar en títulos
        mostrar_grafico : bool, default=True
            Si True, muestra la matriz de confusión
        guardar_grafico : bool, default=False
            Si True, guarda la matriz de confusión como imagen
        ruta_guardado : str, optional
            Ruta donde guardar el gráfico (ej: 'plots/confusion_matrix.png')
        
        Returns:
        --------
        dict : Diccionario con todas las métricas calculadas
        """
        
        # Calcular métricas 
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')    #average = 'weighted' es el promedio ponderado según la cantidad de muestras por clase
        precision_weighted = precision_score(y_test, y_pred, average='weighted')
        recall_weighted = recall_score(y_test, y_pred, average='weighted')
        
        # Calcular ROC-AUC si se proporcionan probabilidades
        roc_auc = None
        if y_pred_proba is not None:
            # Si y_pred_proba tiene 2 columnas, tomar la segunda (clase positiva)
            if y_pred_proba.shape[1] == 2:
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')  #Si no tiene dos columnas, es multiclase. Ponderada. 


        # Imprimir resultados
        print("\n" + "="*60)
        print(f"RESULTADOS EN TEST SET - {nombre_modelo.upper()}")
        print("="*60)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision weighted: {precision_weighted:.4f}")
        print(f"Recall weighted:    {recall_weighted:.4f}")
        print(f"F1-Score weighted:  {f1_weighted:.4f}")
        print(f"ROC-AUC multiclase ponderada:   {roc_auc:.4f}")  #Arma roc_auc para multiclase
        
        print("\n" + "-"*60)
        print("REPORTE DE CLASIFICACIÓN:")
        print("-"*60)
        print(classification_report(y_test, y_pred))
        
        # Matriz de confusión
        if mostrar_grafico or guardar_grafico:
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                        annot_kws={'size': 14})
            
            titulo = f'Matriz de Confusión - {nombre_modelo}\n'
            titulo += f'F1-Score weighted: {f1_weighted:.4f} | Accuracy: {accuracy:.4f}'
            plt.title(titulo, fontsize=12, pad=20)
            plt.ylabel('Valores Reales', fontsize=11)
            plt.xlabel('Predicciones', fontsize=11)
            plt.tight_layout()
            
            if guardar_grafico:
                if ruta_guardado is None:
                    ruta_guardado = f'confusion_matrix_{nombre_modelo.lower().replace(" ", "_")}.png'
                plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
                print(f"\n✅ Gráfico guardado en: {ruta_guardado}")
            
            if mostrar_grafico:
                plt.show()
            else:
                plt.close()
        
        # Retornar diccionario con métricas
        metricas = {
            'Modelo': nombre_modelo,
            'Accuracy': round(accuracy, 4),
            'Precision weighted': round(precision_weighted, 4),
            'Recall weighted': round(recall_weighted, 4),
            'F1_Score weighted': round(f1_weighted, 4),
            'ROC-AUC multiclase ponderada': round(roc_auc, 4)
            }
        
        # Binariza para armar One Vs Rest        
        if y_pred_proba is not None:
            clases = np.unique(y_test)
            y_test_bin = label_binarize(y_test, classes=clases)

        # Graficar todas las curvas ROC en el mismo lienzo
        plt.figure(figsize=(8, 6))

        for i, clase in enumerate(clases):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            auc_clase = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
            plt.plot(fpr, tpr, label=f'{clase} (AUC = {auc_clase:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curvas ROC por clase (One-vs-Rest)')
        plt.legend(title='Clases')
        plt.grid(True)
        plt.show()

        return metricas


    def evaluar_y_guardar_modelo(self, grid_search, X_test, y_test, y_pred, y_pred_proba, nombre_modelo,
                                 results_df=None, mostrar_grafico=True):
        """
        Evalúa un modelo entrenado con GridSearch y guarda resultados completos.
        
        Parameters:
        -----------
        grid_search : GridSearchCV
            Objeto GridSearchCV ya entrenado
        X_test, y_test : array-like
            Datos de test
        nombre_modelo : str
            Nombre del modelo
        results_df : DataFrame, optional
            DataFrame existente para agregar resultados
        mostrar_grafico : bool
            Si mostrar matriz de confusión
        
        Returns:
        --------
        DataFrame con todas las columnas: Modelo, Hiperparametros, Accuracy, 
        F1_Score, Precision, Recall, ROC_AUC, CV_F1_Mean, CV_F1_Std, Fecha
        """
        
        # Predicciones
        y_pred = grid_search.predict(X_test)
        y_pred_proba = grid_search.predict_proba(X_test)
        
        # Evaluar (muestra métricas y gráfico)
        metricas = self._evaluar_modelo(y_test, y_pred, y_pred_proba, 
                                nombre_modelo=nombre_modelo,
                                mostrar_grafico=mostrar_grafico)
        
        # Crear diccionario completo con todas las columnas
        resultado_completo = {
            'Modelo': nombre_modelo,
            'Hiperparametros': str(grid_search.best_params_),
            'Accuracy': metricas['Accuracy'],
            'F1_Score weighted': metricas['F1_Score weighted'],
            'Precision weighted': metricas['Precision weighted'],
            'Recall weighted': metricas['Recall weighted'],
            'ROC-AUC multiclase ponderada': metricas['ROC-AUC multiclase ponderada'],
            'CV_F1_weighted_Mean': round(grid_search.best_score_, 4),
            'CV_F1_weighted_Std': round(grid_search.cv_results_['std_test_score'][grid_search.best_index_], 4),
            'Fecha': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Crear o actualizar DataFrame
        if results_df is None or results_df.empty:
            results_df = pd.DataFrame([resultado_completo])
        else:
            results_df = pd.concat([results_df, pd.DataFrame([resultado_completo])], ignore_index=True)
        
        # Ordenar por F1_Score
        results_df = results_df.sort_values('F1_Score weighted', ascending=False).reset_index(drop=True)
        
        return results_df
    


    def plot_feature_importance(self, modelo, X_train, top_n=50, figsize=(12, 10)):
        """
        Visualiza la importancia de features para modelos de clasificación.
        Soporta modelos directos y Pipelines de sklearn.
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Detectar si es un Pipeline
        if hasattr(modelo, 'named_steps'):
            # Es un Pipeline, extraer el clasificador
            if 'classifier' in modelo.named_steps:
                clasificador = modelo.named_steps['classifier']
            elif 'model' in modelo.named_steps:
                clasificador = modelo.named_steps['model']
            else:
                # Buscar el último paso que sea un clasificador
                clasificador = list(modelo.named_steps.values())[-1]
            print(f"Pipeline detectado. Usando: {type(clasificador).__name__}")
        else:
            clasificador = modelo
        
        feature_names = X_train.columns
        
        # Extraer importancias
        if hasattr(clasificador, 'coef_'):
            coeficientes = clasificador.coef_
            
            if coeficientes.ndim > 1:
                importancia = np.mean(np.abs(coeficientes), axis=0)
                print(f"Modelo multiclase: {len(clasificador.classes_)} clases")
                print(f"Clases: {clasificador.classes_}")
            else:
                importancia = np.abs(coeficientes)
            
            titulo = 'Feature Importance - Logistic Regression'
            xlabel = 'Importancia (|Coeficiente| Promedio)'
            
        elif hasattr(clasificador, 'feature_importances_'):
            importancia = clasificador.feature_importances_
            titulo = 'Feature Importance - Tree-based Model'
            xlabel = 'Importancia'
        else:
            raise ValueError("Modelo no soportado")
        
        # Crear DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importancia
        }).sort_values('importance', ascending=False)
        
        # Visualización
        plt.figure(figsize=figsize)
        top_features = feature_importance_df.head(top_n)
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'], fontsize=9)
        plt.xlabel(xlabel, fontsize=11)
        plt.ylabel('Features', fontsize=11)
        plt.title(f'{titulo}\nTop {top_n} Features', fontsize=13, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()
        
        # Estadísticas
        print(f"\n{'='*60}")
        print(f"Top {min(top_n, len(feature_importance_df))} Features:")
        print(f"{'='*60}")
        print(feature_importance_df.head(top_n).to_string(index=False))
        print(f"\nImportancia acumulada: {top_features['importance'].sum():.4f}")
        
        return feature_importance_df
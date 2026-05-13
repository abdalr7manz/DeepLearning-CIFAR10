import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

# ============================================
# REQUIREMENT 1: DATASET SELECTION AND TECHNICAL SPECIFICATIONS
# ============================================

print("="*70)
print("DATASET: CIFAR-10 | 10 Classes | RGB | 32x32 → 96x96")
print("="*70)

(x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()

x_train = x_train_full[:8000]
y_train = y_train_full[:8000]
x_val = x_train_full[8000:10000]
y_val = y_train_full[8000:10000]
x_test = x_test_full[:2000]
y_test = y_test_full[:2000]

print(f"Training: {x_train.shape[0]} | Validation: {x_val.shape[0]} | Test: {x_test.shape[0]}")

x_train = tf.image.resize(x_train, (96, 96)).numpy()
x_val = tf.image.resize(x_val, (96, 96)).numpy()
x_test = tf.image.resize(x_test, (96, 96)).numpy()

x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.1
)

# ============================================
# REQUIREMENT 2: DL MODEL SELECTION
# ============================================

print("\n" + "="*70)
print("MODEL: MobileNetV1 (Howard et al., 2017)")
print("Architecture: Depthwise Separable Convolutions | 13.4M params")
print("="*70)

# ============================================
# APPROACH 1: DL-BASED FEATURE LEARNING + ML CLASSIFIER
# ============================================

print("\n[APPROACH 1] Feature Extraction + Logistic Regression")

base_model_1 = MobileNet(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
base_model_1.trainable = False

feature_extractor = Model(
    inputs=base_model_1.input,
    outputs=GlobalAveragePooling2D()(base_model_1.output)
)

start1 = time.time()

print("Extracting features from training set...")
train_features = feature_extractor.predict(x_train, verbose=1, batch_size=64)
print("Extracting features from validation set...")
val_features = feature_extractor.predict(x_val, verbose=1, batch_size=64)
print("Extracting features from test set...")
test_features = feature_extractor.predict(x_test, verbose=1, batch_size=64)

y_train_flat = y_train.ravel()
y_val_flat = y_val.ravel()
y_test_flat = y_test.ravel()

print("Training Logistic Regression classifier...")
lr_classifier = LogisticRegression(max_iter=800, random_state=42, C=0.1, solver='lbfgs')
lr_classifier.fit(train_features, y_train_flat)

pred1 = lr_classifier.predict(test_features)

acc1 = accuracy_score(y_test_flat, pred1)
pre1 = precision_score(y_test_flat, pred1, average='weighted')
rec1 = recall_score(y_test_flat, pred1, average='weighted')
f11 = f1_score(y_test_flat, pred1, average='weighted')
time1 = time.time() - start1

print(f"\n✓ Approach 1 Complete: Accuracy={acc1:.4f}, Time={time1:.1f}s")

cm1 = confusion_matrix(y_test_flat, pred1)

# ============================================
# APPROACH 2: END-TO-END DEEP LEARNING MODEL
# ============================================

print("\n[APPROACH 2] End-to-End MobileNet with Fine-Tuning")

base_model_2 = MobileNet(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
base_model_2.trainable = False

model_e2e = Sequential([
    base_model_2,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model_e2e.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

y_train_cat = to_categorical(y_train, 10)
y_val_cat = to_categorical(y_val, 10)
y_test_cat = to_categorical(y_test, 10)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]

start2 = time.time()

print("Phase 1: Training frozen base (3 epochs)...")
history2_phase1 = model_e2e.fit(
    datagen.flow(x_train, y_train_cat, batch_size=64),
    validation_data=(x_val, y_val_cat),
    epochs=3,
    verbose=1,
    callbacks=callbacks
)

print("\nPhase 2: Fine-tuning (3 epochs)...")
base_model_2.trainable = True
for layer in base_model_2.layers[:-15]:
    layer.trainable = False

model_e2e.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2_phase2 = model_e2e.fit(
    datagen.flow(x_train, y_train_cat, batch_size=64),
    validation_data=(x_val, y_val_cat),
    epochs=3,
    verbose=1,
    callbacks=callbacks
)

print("\nPhase 3: Full fine-tuning (2 epochs)...")
for layer in base_model_2.layers:
    layer.trainable = True

model_e2e.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2_phase3 = model_e2e.fit(
    datagen.flow(x_train, y_train_cat, batch_size=64),
    validation_data=(x_val, y_val_cat),
    epochs=2,
    verbose=1,
    callbacks=callbacks
)

pred2_probs = model_e2e.predict(x_test, verbose=1, batch_size=64)
pred2 = np.argmax(pred2_probs, axis=1)

acc2 = accuracy_score(y_test_flat, pred2)
pre2 = precision_score(y_test_flat, pred2, average='weighted')
rec2 = recall_score(y_test_flat, pred2, average='weighted')
f12 = f1_score(y_test_flat, pred2, average='weighted')
time2 = time.time() - start2

print(f"\n✓ Approach 2 Complete: Accuracy={acc2:.4f}, Time={time2:.1f}s")

cm2 = confusion_matrix(y_test_flat, pred2)

# ============================================
# PERFORMANCE SUMMARY TABLE
# ============================================

print("\n" + "="*70)
print("PERFORMANCE COMPARISON SUMMARY")
print("="*70)
print(f"{'Metric':<20} {'Approach 1 (LR)':<25} {'Approach 2 (E2E)':<25}")
print("-"*70)
print(f"{'Accuracy':<20} {acc1:<25.4f} {acc2:<25.4f}")
print(f"{'Precision':<20} {pre1:<25.4f} {pre2:<25.4f}")
print(f"{'Recall':<20} {rec1:<25.4f} {rec2:<25.4f}")
print(f"{'F1-Score':<20} {f11:<25.4f} {f12:<25.4f}")
print(f"{'Training Time (s)':<20} {time1:<25.1f} {time2:<25.1f}")
print(f"{'Improvement':<20} {'':<25} {((acc2-acc1)/acc1*100):.1f}%")
print("="*70)

# ============================================
# CONFUSION MATRICES
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ConfusionMatrixDisplay(cm1).plot(ax=axes[0])
axes[0].set_title('Approach 1: MobileNet + Logistic Regression', fontsize=12, fontweight='bold')

ConfusionMatrixDisplay(cm2).plot(ax=axes[1])
axes[1].set_title('Approach 2: End-to-End MobileNet', fontsize=12, fontweight='bold')

plt.suptitle('Confusion Matrix Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# COMPARISON BAR CHART
# ============================================

metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
approach1_scores = [acc1, pre1, rec1, f11]
approach2_scores = [acc2, pre2, rec2, f12]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))

bars1 = ax.bar(x - width/2, approach1_scores, width, label='Approach 1 (Feature Extractor + LR)', 
               color='#1f77b4', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, approach2_scores, width, label='Approach 2 (End-to-End DL)', 
               color='#ff7f0e', edgecolor='black', linewidth=1.2)

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison: Approach 1 vs Approach 2', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=11)
ax.legend(loc='lower right', fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_facecolor('#f0f0f0')
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('comparison_barchart.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# LEARNING CURVES
# ============================================

total_train_acc = history2_phase1.history['accuracy'] + history2_phase2.history['accuracy'] + history2_phase3.history['accuracy']
total_val_acc = history2_phase1.history['val_accuracy'] + history2_phase2.history['val_accuracy'] + history2_phase3.history['val_accuracy']
total_train_loss = history2_phase1.history['loss'] + history2_phase2.history['loss'] + history2_phase3.history['loss']
total_val_loss = history2_phase1.history['val_loss'] + history2_phase2.history['val_loss'] + history2_phase3.history['val_loss']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, len(total_train_acc)+1), total_train_acc, 'b-', linewidth=2, label='Training Accuracy')
axes[0].plot(range(1, len(total_val_acc)+1), total_val_acc, 'r-', linewidth=2, label='Validation Accuracy')
axes[0].axvline(x=len(history2_phase1.history['accuracy']), color='gray', linestyle='--', linewidth=1.5, label='Phase 1→2')
axes[0].axvline(x=len(history2_phase1.history['accuracy'])+len(history2_phase2.history['accuracy']), 
                color='purple', linestyle='--', linewidth=1.5, label='Phase 2→3')
axes[0].set_title('Learning Curves (Approach 2 - End-to-End)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Accuracy', fontsize=11)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3, linestyle='--')

axes[1].plot(range(1, len(total_train_loss)+1), total_train_loss, 'b-', linewidth=2, label='Training Loss')
axes[1].plot(range(1, len(total_val_loss)+1), total_val_loss, 'r-', linewidth=2, label='Validation Loss')
axes[1].axvline(x=len(history2_phase1.history['loss']), color='gray', linestyle='--', linewidth=1.5, label='Phase 1→2')
axes[1].axvline(x=len(history2_phase1.history['loss'])+len(history2_phase2.history['loss']), 
                color='purple', linestyle='--', linewidth=1.5, label='Phase 2→3')
axes[1].set_title('Loss Curves (Approach 2 - End-to-End)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Loss', fontsize=11)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# SUBPLOT: ALL COMPARISONS
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

axes[0,0].bar(['Approach 1', 'Approach 2'], [acc1, acc2], color=['#1f77b4', '#ff7f0e'])
axes[0,0].set_ylabel('Accuracy')
axes[0,0].set_title('Test Accuracy Comparison')
axes[0,0].set_ylim(0, 1)
for i, v in enumerate([acc1, acc2]):
    axes[0,0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

axes[0,1].bar(['Approach 1', 'Approach 2'], [time1, time2], color=['#1f77b4', '#ff7f0e'])
axes[0,1].set_ylabel('Time (seconds)')
axes[0,1].set_title('Training Time Comparison')
for i, v in enumerate([time1, time2]):
    axes[0,1].text(i, v + 1, f'{v:.1f}s', ha='center', fontweight='bold')

axes[1,0].bar(metrics_names, approach1_scores, color='#1f77b4')
axes[1,0].set_ylabel('Score')
axes[1,0].set_title('Approach 1: All Metrics')
axes[1,0].set_ylim(0, 1)
for i, v in enumerate(approach1_scores):
    axes[1,0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

axes[1,1].bar(metrics_names, approach2_scores, color='#ff7f0e')
axes[1,1].set_ylabel('Score')
axes[1,1].set_title('Approach 2: All Metrics')
axes[1,1].set_ylim(0, 1)
for i, v in enumerate(approach2_scores):
    axes[1,1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.suptitle('Comprehensive Performance Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('comprehensive_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# RADAR CHART
# ============================================

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

metrics_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Speed\n(inverse time)']
approach1_radar = [acc1, pre1, rec1, f11, time2/(time1+time2)]
approach2_radar = [acc2, pre2, rec2, f12, time1/(time1+time2)]

angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
approach1_radar += approach1_radar[:1]
approach2_radar += approach2_radar[:1]
angles += angles[:1]

ax.plot(angles, approach1_radar, 'o-', linewidth=2, label='Approach 1 (LR)', color='#1f77b4')
ax.fill(angles, approach1_radar, alpha=0.25, color='#1f77b4')
ax.plot(angles, approach2_radar, 'o-', linewidth=2, label='Approach 2 (E2E)', color='#ff7f0e')
ax.fill(angles, approach2_radar, alpha=0.25, color='#ff7f0e')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_radar, fontsize=10)
ax.set_ylim(0, 1)
ax.set_title('Multi-Metric Radar Chart Comparison', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig('radar_chart.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# FINAL OUTPUT
# ============================================

print("\n" + "="*70)
print(" PROJECT COMPLETED SUCCESSFULLY")
print("="*70)
print("\n GENERATED FILES:")
print("   1. confusion_matrices.png")
print("   2. comparison_barchart.png")
print("   3. learning_curves.png")
print("   4. comprehensive_analysis.png")
print("   5. radar_chart.png")
print("\n KEY FINDINGS:")
print(f"   • Approach 1 (LR): {acc1:.2%} accuracy in {time1:.1f}s")
print(f"   • Approach 2 (E2E): {acc2:.2%} accuracy in {time2:.1f}s")
print(f"   • Improvement: {((acc2-acc1)/acc1*100):.1f}% better with E2E")
print("="*70)
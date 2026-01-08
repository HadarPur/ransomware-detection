from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def evaluate_model_performance(detector, test_df):
    """
    Implementation for Task 7: Evaluation metrics and per-variant results.
    """
    # Generate predictions
    y_true = test_df['is_encrypted']
    y_pred_labels = detector.evaluate_batch(test_df)

    # Map 'ENCRYPTED' back to 1 and 'NOT ENCRYPTED' to 0 for metrics
    y_pred = y_pred_labels.map({'ENCRYPTED': 1, 'NOT ENCRYPTED': 0})

    # 1. Overall Performance
    print("\n--- Overall Performance ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred))

    # # 2. Per-Variant Results (if variant column exists)
    # if 'variant' in test_df.columns:
    #     print("\n--- Per-Ransomware-Variant Accuracy ---")
    #     for variant in test_df['variant'].unique():
    #         variant_data = test_df[test_df['variant'] == variant]
    #         v_true = variant_data['is_encrypted']
    #         v_pred_labels = detector.evaluate_batch(variant_data)
    #         v_pred = v_pred_labels.map({'ENCRYPTED': 1, 'NOT ENCRYPTED': 0})
    #
    #         acc = accuracy_score(v_true, v_pred)
    #         print(f"Variant {variant}: {acc * 100:.1f}% detection rate")
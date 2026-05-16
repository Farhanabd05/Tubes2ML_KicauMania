def plot_training_history(repo_root=None):
    import matplotlib.pyplot as plt
    from .experiment import load_training_history

    history = load_training_history(repo_root)
    for model_type, group in history.groupby("model_type"):
        plt.figure(figsize=(12, 5))
        for _, row in group.iterrows():
            label = f"{row['variation_name']} L{row['layers']} H{row['hidden_state']}"
            plt.plot(row["history_loss"], label=f"train {label}")
            plt.plot(row["history_val_loss"], linestyle="--", label=f"val {label}")
        plt.title(f"{model_type} Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.show()

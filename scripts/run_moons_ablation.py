from src.config import FIGURES_DIR, get_hyperparams_config
from src.data_utils import DataRepository
from src.models import HiddenLayerClassifier
from src.trainers import HiddenLayerTrainer
from src.plotting import DecisionBoundaryPlotter

def main():
    # 1. Setup Data
    repo = DataRepository()
    X_train, y_train, X_val, y_val, X_test, y_test = repo.load_dataset("moons")
    X_train, X_val, X_test = repo.standardize_splits(X_train, X_val, X_test)
    
    plotter = DecisionBoundaryPlotter()
    results = []

    # 2. Run your Ablation Loop
    for width in [2, 8, 32]:
        print(f"Training width {width}...")
        
        # Initialize Model using the class you shared
        model = HiddenLayerClassifier(input_dim=2, hidden_dim=width, output_dim=2)
        model.initialize(seed=42)
        
        # Get defaults from config and override as needed
        params = get_hyperparams_config("moons", "hidden_layer")
        trainer = HiddenLayerTrainer(
            epochs=1200, 
            learning_rate=0.3, 
            lambda_reg=0.01,
            batch_size=params["batch_size"]
        )
        
        # Train
        trainer.fit(model, X_train, y_train, X_val, y_val)
        
        # Evaluate
        loss_val, acc_val = model.evaluate(X_val, y_val)
        loss_test, acc_test = model.evaluate(X_test, y_test)
        
        # Plot (Saves directly to main/figures/)
        plotter.plot_decision_boundary(
            X_test, y_test, model.predict, 
            FIGURES_DIR / f"width_{width}.png", 
            f"Ablation: Width {width}"
        )
        
        results.append({
            "width": width, "val_ce": loss_val, "val_acc": acc_val,
            "test_ce": loss_test, "test_acc": acc_test
        })

    # 3. Print the table (exact same logic as yours)
    print("\n" + "="*70)
    print(f"{'Width':<8} | {'Val CE':<10} | {'Val Acc':<10} | {'Test CE':<10} | {'Test Acc':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['width']:<8} | {r['val_ce']:<10.4f} | {r['val_acc']:<10.2%} | "
              f"{r['test_ce']:<10.4f} | {r['test_acc']:<10.2%}")

if __name__ == "__main__":
    main()

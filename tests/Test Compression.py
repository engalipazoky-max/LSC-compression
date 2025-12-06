from lsc import EnhancedLSCEngine, LSCConfig

# Load a sample attention matrix
att = torch.load("demo_data/attention_matrices/layer_3/head_4.pt")

# Initialize LSC engine
config = LSCConfig(compression_method='hybrid')
engine = EnhancedLSCEngine(config)

# Compress
compressed = engine.compress(att.unsqueeze(0).unsqueeze(0))  # Add batch/head dims

# Compare with expected results
expected = dataset['compression_results']['layer_3']['head_4']['hybrid']
print(f"MSE: {expected['mse']:.6f}")
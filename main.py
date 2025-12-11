import yaml

from src.pipeline import Pipeline

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    pipeline = Pipeline(config)
    pipeline.execute()

import wandb

class WandbRecorder:
    def __init__(self, project_name, config, run_name=None):
        # Initialize the W&B run
        wandb.init(project=project_name, config=config, name=run_name)

    def record_curve(self, name, data, curr_iter):
        if isinstance(data, (tuple, list)):
            for idx, data_item in enumerate(data):
                wandb.log({f"{name}_{idx}": data_item, "iteration": curr_iter})
        else:
            wandb.log({name: data, "iteration": curr_iter})

    def record_image(self, name, data, curr_iter):
        # Log the image using W&B's image support
        wandb.log({name: [wandb.Image(data)], "iteration": curr_iter})

    def record_images(self, data_container: dict, curr_iter):
        # Log a batch of images
        for name, data in data_container.items():
            wandb.log({name: [wandb.Image(data)], "iteration": curr_iter})

    def record_histogram(self, name, data, curr_iter):
        # W&B automatically logs histograms when tensors are logged
        wandb.log({name: data, "iteration": curr_iter})

    def close(self):
        wandb.finish()

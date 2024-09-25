import wandb

class WandbRecorder:
    def __init__(self, project_name, config, run_name=None):
        # Initialize the W&B run
        wandb.init(project=project_name, config=config, name=run_name)

    def record_curve(self, name, data, curr_epoch):
        if isinstance(data, (tuple, list)):
            for idx, data_item in enumerate(data):
                wandb.log({f"{name}_{idx}": data_item, "epoch": curr_epoch}, step=curr_epoch)
        else:
            wandb.log({name: data, "epoch": curr_epoch}, step=curr_epoch)

    def record_val_curve(self, name, data, curr_epoch):
        if isinstance(data, (tuple, list)):
            for idx, data_item in enumerate(data):
                wandb.log({f"{name}_{idx}": data_item, "epoch": curr_epoch}, step=curr_epoch)
        else:
            wandb.log({name: data, "epoch": curr_epoch}, step=curr_epoch)

    def record_image(self, name, data, curr_epoch):
        # Log the image using W&B's image support
        wandb.log({name: [wandb.Image(data)], "epoch": curr_epoch}, step=curr_epoch)

    def record_images(self, data_container: dict, curr_epoch):
        # Log a batch of images
        for name, data in data_container.items():
            wandb.log({name: [wandb.Image(data)], "epoch": curr_epoch}, step=curr_epoch)

    def record_prediction_table(self, epoch, images, masks, sals):
        """
        Logs a W&B table that tracks the model predictions over different epochs.
        
        Args:
            epoch (int): The current epoch.
            images (torch.Tensor): The input images (B, C, H, W).
            masks (torch.Tensor): The ground truth masks (B, H, W).
            sals (torch.Tensor): Saliency maps (B, H, W).
        """
        # Initialize the table with columns
        table = wandb.Table(columns=["Epoch", "Image", "Mask", "Saliency Map"])

        # Add rows to the table with image, mask, and saliency map for each data point
        for img, mask, sal in zip(images, masks, sals):
            table.add_data(epoch, wandb.Image(img), wandb.Image(mask), wandb.Image(sal))

        # Log the table to W&B
        wandb.log({f"Predictions at Epoch": table})

    def record_histogram(self, name, data, curr_epoch):
        # W&B automatically logs histograms when tensors are logged
        wandb.log({name: data, "epoch": curr_epoch})

    def record_epoch_metrics(self, metrics_dict, curr_epoch):
        """
        Logs epoch-level metrics to W&B. It logs all metrics in the provided dictionary 
        with the current epoch as the step.

        Args:
            metrics_dict (dict): A dictionary containing metric names and values.
            curr_epoch (int): The current epoch.
        """
        # Log metrics to W&B
        wandb.log(metrics_dict, step=curr_epoch)

    def close(self):
        wandb.finish()

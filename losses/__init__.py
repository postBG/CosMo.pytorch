from losses.batch_based_classification_loss import BatchBasedClassificationLoss


def loss_factory(config):
    return {
        'metric_loss': BatchBasedClassificationLoss(),
    }

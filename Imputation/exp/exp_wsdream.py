from exp.exp_imputation import Exp_Imputation as BaseExp


class Exp_Imputation(BaseExp):
    def prepare_batch(self, batch_x):
        batch_x = super().prepare_batch(batch_x)
        return batch_x.permute(0, 2, 1)

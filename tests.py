import unittest
import torch

from concrete_autoencoder import ConcreteAutoEncoder

class TestCAMissing(unittest.TestCase):
    def setUp(self):
        self.input = torch.randn(4, 1, 10)
        mask_idx = torch.randint(0, 10, (4,))
        self.mask = torch.ones_like(self.input)
        for i, idx in enumerate(mask_idx):
            self.mask[i, :, idx] = 0
        self.inv_mask = self.mask.int() ^ 1

        self.model = ConcreteAutoEncoder(10, 5, decoder_type='mlp')

    @torch.no_grad()
    def test_encoder_masking(self):
        X, selection = self.model.encoder(self.input, train=True, X_mask=self.mask, debug=True)

        self.assertAlmostEqual(0., torch.sum(X - self.input * self.mask))
        self.assertEqual(0., torch.sum(selection * self.inv_mask))

    @torch.no_grad()
    def test_output_masking(self):
        out = self.model(self.input, train=True, X_mask=self.mask)

        self.assertEqual(0., torch.sum(out * self.inv_mask))

    def test_grad_masking(self):
        x = self.input[0].unsqueeze(0)
        x_mask = self.mask[0].unsqueeze(0)
        out = self.model(x, train=True, X_mask=x_mask)

        loss = torch.nn.functional.mse_loss(out, x)
        loss.backward()

        encoder_grad = self.model.encoder.logits.grad

        self.assertGreaterEqual(1e-5, torch.sum((encoder_grad * self.inv_mask)**2))

if __name__ == "__main__":
    unittest.main()


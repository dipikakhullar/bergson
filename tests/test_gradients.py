import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    GradientCollector,
    GradientProcessor,
    ProjectionGenerator,
)


def test_phi3():
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM")
    model = AutoModelForCausalLM.from_config(config)

    # It's important that we use a batch size of one so that we can simply use the
    # aggregate gradients from the backward itself and compare against those
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], device=model.device)
    inputs = dict(input_ids=tokens, labels=tokens)

    # Test with 16 x 16 random projection as well as with no projection
    for p in (16, None):
        processor = GradientProcessor(projection_dim=p)
        with GradientCollector(model, processor) as collector:
            model.zero_grad()
            model(**inputs).loss.backward()

        adafactors: dict[str, AdafactorNormalizer] = {}
        adams: dict[str, AdamNormalizer] = {}

        # Go through the motions of what GradientCollector does, but after the fact
        generator = ProjectionGenerator(model.device, processor.projection_seed)
        for name, layer in model.named_modules():
            if not isinstance(layer, nn.Linear):
                continue

            o, i = layer.out_features, layer.in_features
            g = layer.weight.grad
            assert g is not None

            moments = g.square()
            if p is not None:
                A, B = generator.next_projection(p, p, o, i)
                g = A @ g @ B.T

            collected_grad = collector.collected_grads[name].squeeze(0)
            torch.testing.assert_close(g, collected_grad)

            # Store normalizers for this layer
            adams[name] = AdamNormalizer(moments)
            adafactors[name] = adams[name].to_adafactor()

        # Now do it again but this time use the normalizers
        for normalizers in (adafactors, adams):
            processor = GradientProcessor(normalizers=normalizers, projection_dim=p)
            with GradientCollector(model, processor) as collector:
                model.zero_grad()
                model(**inputs).loss.backward()

            generator = ProjectionGenerator(model.device, processor.projection_seed)
            for name, layer in model.named_modules():
                if not isinstance(layer, nn.Linear):
                    continue

                o, i = layer.out_features, layer.in_features
                g = layer.weight.grad
                assert g is not None

                g = normalizers[name].normalize_(g)
                if p is not None:
                    A, B = generator.next_projection(p, p, o, i)
                    g = A @ g @ B.T

                # Compare the normalized gradient with the collected gradient. We use a
                # higher tolerance than the default because there seems to be some
                # non-negligible numerical error that accumulates due to the different
                # order of operations. Maybe we should look into this more.
                collected_grad = collector.collected_grads[name].squeeze(0)
                torch.testing.assert_close(g, collected_grad, atol=1e-4, rtol=1e-4)

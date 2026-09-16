"""One initialization factory for three independent candidates and a reference."""

import torch

from experiments.sync_delta_v1.models import build_model as build_reference

VARIANTS = ("routing_sync", "direct_sync", "feature_sync")
ALL_VARIANTS = (*VARIANTS, "sync_delta")


def build_model(num_flows, variant, init_seed=41001):
    from .routing import RoutingSyncModel
    from .direct import DirectSyncModel
    from .feature import FeatureSyncModel

    reference = build_reference(num_flows, init_seed=int(init_seed))
    if variant == "sync_delta":
        return reference
    classes = {"routing_sync": RoutingSyncModel, "direct_sync": DirectSyncModel,
               "feature_sync": FeatureSyncModel}
    if variant not in classes:
        raise ValueError(f"Unknown candidate: {variant}")
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device="cpu").manual_seed(int(init_seed) + 91417)
        torch.random.set_rng_state(generator.get_state())
        model = classes[variant](num_flows)
    missing, unexpected = model.load_state_dict(reference.state_dict(), strict=False)
    if unexpected or any(name in reference.state_dict() for name in missing):
        raise RuntimeError("Candidate does not preserve all shared parameter identities")
    for name, value in reference.state_dict().items():
        if not torch.equal(model.state_dict()[name], value):
            raise RuntimeError(f"Shared initialization differs: {name}")
    return model

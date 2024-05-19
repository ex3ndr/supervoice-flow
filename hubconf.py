dependencies = ['torch', 'torchaudio']

def flow():
    import torch
    from supervoice_flow.model import AudioFlow
    from supervoice_flow.config import config
    model = AudioFlow(config)
    checkpoint = torch.hub.load_state_dict_from_url("https://shared.korshakov.com/models/supervoice_flow.pt", map_location="cpu")
    model.load_state_dict(checkpoint['model'])
    return model
            
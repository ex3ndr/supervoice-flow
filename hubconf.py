dependencies = ['torch', 'torchaudio']

def flow():
    import torch
    from supervoice_flow.model import AudioFlow
    model = AudioFlow()
    checkpoint = torch.hub.load_state_dict_from_url("https://shared.korshakov.com/models/supervoice_flow.pt")
    model.load_state_dict(checkpoint['model'])
    return model
            
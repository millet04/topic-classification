import torch
import numpy as np

def get_accuracy(logits, labels):
    logits_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = labels.flatten() # not essential
    return (np.sum(logits_flat == labels_flat) / len(labels_flat))*100

def test(model, test_dataloader, device):
    
    total_accuracy = 0
    
    model.eval()    
    for _, batch in enumerate(test_dataloader):
            
        encoder_input_ids = batch['encoder_input_ids'].to(device)
        encoder_attention_mask = batch['encoder_attention_mask'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        decoder_attention_mask = batch['decoder_attention_mask'].to(device)        
        
        label = batch['label'].to(device)
            
        with torch.no_grad():
            logits = model(encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask)
        
        logits = logits.detach().cpu().numpy()
        label = label.to('cpu').numpy()
        accuracy = get_accuracy(logits, label)
        total_accuracy += accuracy
    
    avg_accuracy = total_accuracy / len(test_dataloader)
    return avg_accuracy
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
            
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        label = batch['label'].to(device)
            
        with torch.no_grad():
            logits = model(input_ids, attention_mask, token_type_ids)
        
        logits = logits.detach().cpu().numpy()
        label = label.to('cpu').numpy()
        accuracy = get_accuracy(logits, label)
        total_accuracy += accuracy
    
    avg_accuracy = total_accuracy / len(test_dataloader)
    return avg_accuracy
import torch
import numpy as np

def get_accuracy(logits, labels):
    logits_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = np.array(labels).flatten() # not essential
    return (np.sum(logits_flat == labels_flat) / len(labels_flat)) * 100

def test(model, test_dataloader, num_labels, device):
    
    total_accuracy = 0   
    
    model.eval()    
    for _, batch in enumerate(test_dataloader):
            
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        label = batch['label']
            
        with torch.no_grad():
            logits = model(input_ids, attention_mask, token_type_ids)
        
        # (batch_size * num_labels, 2)
        logits = logits.detach().cpu().numpy()

        # (batch_size, num_labels, 2)
        logits = logits.reshape(-1, num_labels, 2)
        
        # Get label with max logits for entailment (0)
        # (batch_size, num_labels)
        logits = logits[:, :, 0]
        
        accuracy = get_accuracy(logits, label)
        total_accuracy += accuracy
    
    avg_accuracy = total_accuracy / len(test_dataloader)
    return avg_accuracy      
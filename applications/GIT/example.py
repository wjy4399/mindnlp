##minspore的写法

from mindnlp.transformers import LayoutLMv2ForTokenClassification

num_labels = len(label2count.keys())
model = LayoutLMv2ForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
from mindnlp.core.optim import AdamW
optimizer = AdamW(model.trainable_params(), lr =5e-5)

global_step = 0
num_train_epochs = 4
from mindnlp.core.autograd import value_and_grad

def forward_fn(batch):
    # get the inputs;
    input_ids = batch['input_ids']
    bbox = batch['bbox']
    image = batch['image']
    attention_mask = batch['attention_mask']
    token_type_ids = batch['token_type_ids']
    labels = batch['labels']

    outputs = model(input_ids=input_ids,
                    bbox=bbox,
                    image=image,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels) 
    loss = outputs.loss
    
    return loss

grad_fn = value_and_grad(forward_fn, model.trainable_params(), attach_grads=True)
from tqdm import tqdm

# put the model in training mode
model.set_train(True)

for epoch in range(num_train_epochs):  
    print("Epoch:", epoch)
    for batch in tqdm(train_dataloader.create_dict_iterator()):
        optimizer.zero_grad()
        # forward, backward + optimize
        loss = grad_fn(batch)
        optimizer.step()


        # print loss every 100 steps
        if global_step % 100 == 0:
            print(f"Loss after {global_step} steps: {loss.item()}")

        global_step += 1
model.save_pretrained("./LayoutLMv2_For_TokenClassificaion/Checkpoints202411")
processor.save_pretrained('./LayoutLMv2_For_TokenClassificaion/processorPretrained')

#torch 的写法


from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2)


from transformers import LayoutLMv2ForTokenClassification, AdamW
import torch
from tqdm.notebook import tqdm

model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased',
                                                                      num_labels=len(labels))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

global_step = 0
num_train_epochs = 4

#put the model in training mode
model.train() 
for epoch in range(num_train_epochs):  
   print("Epoch:", epoch)
   for batch in tqdm(train_dataloader):
        # get the inputs;
        input_ids = batch['input_ids'].to(device)
        bbox = batch['bbox'].to(device)
        image = batch['image'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model(input_ids=input_ids,
                        bbox=bbox,
                        image=image,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels) 
        loss = outputs.loss
        
        # print loss every 100 steps
        if global_step % 100 == 0:
          print(f"Loss after {global_step} steps: {loss.item()}")

        loss.backward()
        optimizer.step()
        global_step += 1

model.save_pretrained("/content/drive/MyDrive/LayoutLMv2/Tutorial notebooks/CORD/Checkpoints")
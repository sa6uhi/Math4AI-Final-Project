import numpy as np


with np.load(r'C:\Users\User\Downloads\moons.npz') as data:
    X_train = data['X_train']
    y_train=data['y_train']

    X_val=data['X_val']
    y_val=data['y_val']

    X_test=data['X_test']
    y_test=data['y_test']

#Consistent scaling
mean=np.mean(X_train,axis=0)
std=np.std(X_train,axis=0)

X_train=(X_train-mean)/std
X_val=(X_val-mean)/std
X_test=(X_test-mean)/std

n_samples=X_train.shape[0]
input_dim=2
hidden_dim=10
output_dim=2
lambda_reg=0.01

#Xavier initialization for tanh
W1=np.random.randn(input_dim,hidden_dim)*np.sqrt(1/input_dim)
b1=np.zeros((1,hidden_dim))

W2=np.random.randn(hidden_dim,output_dim)*np.sqrt(1/output_dim)
b2=np.zeros((1,output_dim))

print(f"Loaded {n_samples} samples.")

#z1=np.dot(X_train,W1)+b1
#h=np.tanh(z1)

#logits=np.dot(h,W2)+b2
#exp_logits=np.exp(logits-np.max(logits,axis=1,keepdims=True))
#probabilities=exp_logits/np.sum(exp_logits,axis=1,keepdims=True)

#dZ2=probabilities.copy()
#dZ2[np.arange(n_samples),y_train]-=1
#dZ2/=n_samples

#dW2=np.dot(h.T,dZ2)+lambda_reg*W2
#db2=np.sum(dZ2,axis=0,keepdims=True)

#dh=np.dot(dZ2,W2.T)
#dZ1=dh*(1-np.power(h,2))

#dW1=np.dot(X_train.T,dZ1)+lambda_reg*W1
#db1=np.sum(dZ1,axis=0,keepdims=True)

eta=0.3 #Learning rate

#W1-=eta*dW1
#b1-=eta*db1
#W2-=eta*dW2
#b2-=eta*db2

epochs=1200


##Step 7

for epoch in range(epochs):
    #Forward pass
    #Layer 1
    z1 = np.dot(X_train, W1) + b1
    h = np.tanh(z1)

    #Layer 2
    logits = np.dot(h, W2) + b2
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    #Loss
    correct_class_probs=probabilities[np.arange(n_samples),y_train]
    log_probs = np.log(correct_class_probs + 1e-15)
    loss_reg=0.5*lambda_reg*(np.sum(np.square(W1))+np.sum(np.square(W2)))
    loss = -np.mean(log_probs)+loss_reg

    #Backpropogation
    #Output layer gradient
    dZ2 = probabilities.copy()
    dZ2[np.arange(n_samples), y_train] -= 1
    dZ2 /= n_samples

    dW2 = np.dot(h.T, dZ2) + lambda_reg * W2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    #Hidden layer gradient
    dh = np.dot(dZ2, W2.T)
    dZ1 = dh * (1 - np.power(h, 2))

    dW1 = np.dot(X_train.T, dZ1) + lambda_reg * W1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    #Update parameters
    W1 -= eta * dW1
    b1 -= eta * db1
    W2 -= eta * dW2
    b2 -= eta * db2

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss {loss:.4f}")
        #Forward pass validation
        z1_val=np.dot(X_val,W1)+b1
        h_val=np.tanh(z1_val)
        logits_val=np.dot(h_val, W2) + b2

        #Loss Validation
        exp_val=np.exp(logits_val-np.max(logits_val,axis=1,keepdims=True))
        probabilities_val=exp_val/np.sum(exp_val,axis=1,keepdims=True)

        correct_class_probs_val=probabilities_val[np.arange(X_val.shape[0]),y_val]
        log_probs_val = np.log(correct_class_probs_val + 1e-15)
        loss_val = -np.mean(log_probs_val)

        #Accuracy
        preds_val=np.argmax(probabilities_val,axis=1)
        accuracy=np.mean(np.equal(preds_val,y_val))

        print(f"Epoch {epoch}: Validation Loss {loss_val:.4f}, Accuracy {accuracy*100:.2f}%")

#Step 9 Testing
z1_test=np.dot(X_test,W1)+b1
h_test=np.tanh(z1_test)
logits_test=np.dot(h_test,W2)+b2
exp_test=np.exp(logits_test-np.max(logits_test,axis=1,keepdims=True))
probabilities_test = exp_test / np.sum(exp_test, axis=1, keepdims=True)

preds_test=np.argmax(probabilities_test,axis=1)
accuracy_final=np.mean(np.equal(preds_test,y_test))

print("-" * 30)
print(f"FINAL TEST RESULTS")
print(f"ACCURACY: {accuracy_final*100:.2f}%")
print("-" * 30)


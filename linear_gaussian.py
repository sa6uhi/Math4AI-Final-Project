import numpy as np

with np.load(r'C:\Users\User\Downloads\linear_gaussian.npz') as data:
    X_train = data['X_train']
    y_train=data['y_train']

    X_val=data['X_val']
    y_val=data['y_val']

    X_test=data['X_test']
    y_test=data['y_test']

X_train=(X_train-np.mean(X_train,axis=0))/np.std(X_train,axis=0)

n_samples=X_train.shape[0]
n_features=2
n_classes=2
W=np.random.randn(n_features,n_classes)*0.1
b=np.zeros((1,n_classes))

print(f"Loaded {n_samples} samples.")

##logits=np.dot(X,W)+b

#def softmax(z):
    #exp_z=np.exp(z-np.max(z,axis=1,keepdims=True))
    #return exp_z/np.sum(exp_z,axis=1,keepdims=True)

#probabilities= softmax(logits)
#print(probabilities)
#correct_class_probs=probabilities[np.arange(n_samples),y]
#log_probs=np.log(correct_class_probs+1e-15)
#loss=-np.mean(log_probs)

#print(f"Loss: {loss}")

##Step 5

#dZ=probabilities.copy()
#dZ[np.arange(n_samples),y]-=1
#dZ/=n_samples

#dW=np.dot(X.transpose(),dZ)

#db=np.sum(dZ,axis=0,keepdims=True)

#print(db)

##Step 6 Update parameters

eta=0.1 #Learning rate

#W-=eta*dW
#b-=eta*db

epochs=200


##Step 7
lambda_reg=0.01

for epoch in range(epochs):
    logits = np.dot(X_train, W) + b

    exp_z=np.exp(logits-np.max(logits,axis=1,keepdims=True))
    probabilities=exp_z/np.sum(exp_z,axis=1,keepdims=True)

    correct_class_probs=probabilities[np.arange(n_samples),y_train]
    log_probs = np.log(correct_class_probs + 1e-15)

    loss_reg=0.5*lambda_reg*np.sum(np.square(W))
    loss = -np.mean(log_probs)+loss_reg

    dZ = probabilities.copy()
    dZ[np.arange(n_samples),y_train]-=1
    dZ/=n_samples

    dW = np.dot(X_train.transpose(), dZ)+lambda_reg*W
    db = np.sum(dZ, axis=0, keepdims=True)

    W-=eta*dW
    b-=eta*db

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss {loss:.4f}")
        logits_val=np.dot(X_val, W) + b
        exp_val=np.exp(logits_val-np.max(logits_val,axis=1,keepdims=True))
        probabilities_val=exp_val/np.sum(exp_val,axis=1,keepdims=True)

        correct_class_probs_val=probabilities_val[np.arange(X_val.shape[0]),y_val]
        log_probs_val = np.log(correct_class_probs_val + 1e-15)
        loss_val = -np.mean(log_probs_val)

        preds_val=np.argmax(probabilities_val,axis=1)
        accuracy=np.mean(np.equal(preds_val,y_val))

        print(f"Epoch {epoch}: Validation Loss {loss_val:.4f}, Accuracy {accuracy*100:.2f}%")

#Step 9 Testing

logits_test=np.dot(X_test,W)+b
exp_test=np.exp(logits_test-np.max(logits_test,axis=1,keepdims=True))
probabilities_test = exp_test / np.sum(exp_test, axis=1, keepdims=True)

preds_test=np.argmax(probabilities_test,axis=1)

accuracy_final=np.mean(np.equal(preds_test,y_test))

print("-" * 30)
print(f"FINAL TEST RESULTS")
print(f"ACCURACY: {accuracy_final*100:.2f}%")
print("-" * 30)


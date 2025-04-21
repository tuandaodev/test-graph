import scipy.sparse as sp
import torch
import scipy.io as scio
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from scipy.sparse.linalg import inv
from sklearn.model_selection import train_test_split
import cv2
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    with np.errstate(divide='ignore'):
        r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def extract_frames(video_path, num_frames=30):
    """Extract frames from video file"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = total_frames // num_frames
    
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if len(frames) == num_frames:
                break
    
    cap.release()
    return np.array(frames)

def create_scene_graph(frames):
    """Create scene graph from video frames"""
    # Initialize ResNet model for feature extraction
    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])  # Remove last FC layer
    model.eval()
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Extract features for each frame
    features = []
    for frame in frames:
        frame_tensor = transform(frame).unsqueeze(0)
        with torch.no_grad():
            feature = model(frame_tensor)
        features.append(feature.squeeze().numpy())
    
    features = np.array(features)
    
    # Create adjacency matrix based on feature similarity
    adj_matrix = np.zeros((len(frames), len(frames)))
    for i in range(len(frames)):
        for j in range(len(frames)):
            if i != j:
                similarity = np.dot(features[i], features[j]) / (np.linalg.norm(features[i]) * np.linalg.norm(features[j]))
                adj_matrix[i, j] = similarity
    
    return features, sp.csr_matrix(adj_matrix)

def load_video_data(video_path, mode='s'):
    """Load video data and create scene graph"""
    print('Loading video data from {}...'.format(video_path))
    
    # Extract frames from video
    frames = extract_frames(video_path)
    
    # Create scene graph
    features, adj = create_scene_graph(frames)
    
    # Create dummy labels (can be modified based on specific task)
    labels = np.zeros((len(frames), 2))
    labels[:len(frames)//2, 0] = 1  # First half normal
    labels[len(frames)//2:, 1] = 1  # Second half anomaly
    
    # Convert to proper format
    adj = compute_laplacian(adj, mode)
    idx_train, idx_test, idx_val, features, labels, adj, minority_label = split_data(features, labels, adj)
    
    gamma = 0.9
    patience = 10
    
    return adj, features, labels, idx_train, idx_val, idx_test, gamma, patience, minority_label

def split_data(features, labels, adj):
    label_counts = np.sum(labels, axis=0)
    minority_label = np.argmin(label_counts)
    majority_label = 1 - minority_label

    minority_indices = np.where(labels[:, minority_label] == 1)[0]
    majority_indices = np.where(labels[:, majority_label] == 1)[0]

    minority_num = len(minority_indices)
    majority_num = minority_num * 2

    minority_indices = np.random.choice(minority_indices, size=minority_num, replace=False)
    majority_indices = np.random.choice(majority_indices, size=majority_num, replace=False)

    selected_indices = np.concatenate([minority_indices, majority_indices])

    selected_features = features[selected_indices]
    selected_labels = labels[selected_indices]

    x_train, x_test_val, y_train, y_test_val = train_test_split(
        selected_features, selected_labels, train_size=0.2, test_size=0.8, stratify=selected_labels
    )

    x_test, x_val, y_test, y_val = train_test_split(
        x_test_val, y_test_val, train_size=0.5, test_size=0.5, stratify=y_test_val
    )

    list_split = list(range(adj.shape[0]))

    node_perm = np.random.permutation(labels.shape[0])
    idx_train = np.where(np.isin(list_split, np.where(np.isin(features, x_train).all(axis=1))[0]))[0]
    idx_val = np.where(np.isin(list_split, np.where(np.isin(features, x_val).all(axis=1))[0]))[0]
    idx_test = np.where(np.isin(list_split, np.where(np.isin(features, x_test).all(axis=1))[0]))[0]

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return idx_train, idx_test, idx_val, features, labels, adj, minority_label

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def compute_laplacian(adj, mode):
    adj = adj.maximum(adj.T)  # Element-wise maximum to make the matrix symmetric
    adj = adj + sp.eye(adj.shape[0])  # Add self-loops by adding identity matrix

    D = sp.diags(adj.sum(axis=1).A.ravel())  # Diagonal matrix with node degrees

    print("adj:", type(adj))
    print("D:", type(D))

    if mode == 's':
        D_sqrt_inv = inv(D).sqrt()  # Compute D^(-0.5) as a sparse matrix
        l = D - adj  # Compute Laplacian
        adj = D_sqrt_inv.dot(l).dot(D_sqrt_inv)
    elif mode == 'r':
        D_inv = inv(D)  # Compute D^(-1) as a sparse matrix
        l = D - adj  # Compute Laplacian
        adj = D_inv.dot(l)
    return(adj)
  
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)  # Predicted labels
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    acc = accuracy_score(labels_np, preds_np)
    auc_roc = roc_auc_score(labels_np, preds_np)
    auc_pr = average_precision_score(labels_np, preds_np)

    return acc, auc_roc, auc_pr

def load_DBLP(dataset="DBLP", mode='s'):
    print('Loading {} dataset...'.format(dataset))
    dataFile = '/content/test-graph/data/' + dataset + '/' + dataset + '.mat'
    data = scio.loadmat(dataFile)
    labels, features = data['label'], data['features'].astype(float)
    N = features.shape[0]
    Networks = np.array([data['net_APA'] - np.eye(N)])
    Networks = np.squeeze(Networks, axis=0)
    Network = sp.csc_matrix(Networks)

    adj = sp.csr_matrix(Network.toarray()[:, :])
    labels = encode_onehot(list(data['label'][:, 0]))
    print('Dataset has {} nodes, {} features.'.format(adj.shape[0], features.shape[1]))
    adj = compute_laplacian(adj,mode)
    idx_train, idx_test, idx_val, features, labels, adj, minority_label = split_data(features, labels,adj)
    gamma = 0.9
    patience = 10

    return adj, features, labels, idx_train, idx_val, idx_test, gamma, patience, minority_label

def load_Yelp(dataset="Yelp", mode='s'):
    print('Loading {} dataset...'.format(dataset))
    dataFile = '/content/test-graph/data/' + dataset + '/' + dataset + '.mat'
    data = scio.loadmat(dataFile)
    labels = np.transpose(data['label'])
    labels = encode_onehot(list(labels[:, 0]))
    features = data['features'].toarray()[:, :]
    adj = sp.csr_matrix(data['homo'].toarray()[:, :])
    print('Dataset has {} nodes, {} features.'.format(adj.shape[0], features.shape[1]))
    adj = compute_laplacian(adj,mode)
    idx_train, idx_test, idx_val, features, labels, adj, minority_label = split_data(features, labels,adj)
    gamma = 0.9
    patience = 10
    return adj, features, labels, idx_train, idx_val, idx_test, gamma, patience, minority_label

def load_Elliptic(dataset="Elliptic", mode='s'):
    print('Loading {} dataset...'.format(dataset))
    dataFile = '/content/test-graph/data/' + dataset + '/' + dataset + '.mat'
    data = scio.loadmat(dataFile)
    edges = data['edge_index']
    labels = np.transpose(data['labels'])
    labels = encode_onehot(list(labels[:, 0]))
    feat_data = data['features']
    num_nodes = 46564
    num_samples = int(num_nodes * 1)
    sampled_indices = np.random.choice(num_nodes, num_samples, replace=False)
    sampled_labels = labels[sampled_indices]
    sampled_feat_data = feat_data[sampled_indices]
    row_indices = np.repeat(sampled_indices, sampled_feat_data.shape[1])
    col_indices = np.tile(np.arange(sampled_feat_data.shape[1]), num_samples)
    values = sampled_feat_data.flatten()
    features_sparse = sp.csr_matrix((values, (row_indices, col_indices)), shape=(num_nodes, sampled_feat_data.shape[1]))
    features = torch.FloatTensor(features_sparse.toarray())
    adj = sp.csr_matrix((np.ones(edges.shape[1]), edges), shape=(num_nodes, num_nodes))
    adj = adj.tocsc()
    print('Dataset has {} nodes, {} features, {} labels.'.format(adj.shape[0], features.shape[1], labels.shape))
    adj = compute_laplacian(adj,mode)
    idx_train, idx_test, idx_val, features, labels, adj, minority_label = split_data(features, labels,adj)
    gamma = 0.9
    patience = 10

    return adj, features, labels, idx_train, idx_val, idx_test, gamma, patience, minority_label

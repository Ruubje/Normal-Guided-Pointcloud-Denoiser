# Created by Yuefan Shen, Jhonve https://github.com/Jhonve/GCN-Denoiser
# Altered by Ruben Band

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def knn(x, k):
    # x: (Batch_index, Feature_index, Point_index)
    # k: ([8]) K nearest neighbors
    # Transpose x to shape x: (Batch_index, Point_index, Feature_index)
    # Matmul with batched matrix x batched matrix to get x: (Batch_index, Point_index, Feature_index) x (Batch_index, Feature_index, Point_index) = (Batch_index, Point_index, Point_index)
    # Multiply by -2
    # Intuition: Inner product of feature vector, given 2 graph points, representing the 'distance' between features of points.
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    # Calculate the sum of squares, but keep the dimensions.
    # xx: (Batch_index, 1, Point_index)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    # The pairwise distance is not actually a distance measure. (The numbers are negative) However, the values can be used to find knn.
    # pairwise_distance = -(Batch_index, 1, Point_index) + 2*(Batch_index, Point_index, Point_index) - (Batch_index, Point_index, 1) = (Batch_index, Point_index, Point_index)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    # Find topk on last dimension (point_index) (second dimension could also be chosen since the matrix is symmetrical)
    # k[0] = 8, since k = [8].
    # topk returns (values, indices), so we select the indices.
    # idx: (Batch_index, Point_index, TopK_index)
    idx = pairwise_distance.topk(k=k[0], dim=-1)[1]
    return idx

def get_graph_feature(x, k):
    # x: (Batch_index, Feature_index, Point_index) -> Feature_value
    # k: (1) -> 8 K nearest neighbors
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points) # Still don't know why you do this? This doesn't change anything?
    idx = knn(x, k=k) # idx: (Batch_index, Point_index, TopK_index) -> Point_index
    # device = torch.device('cuda')
    device = x.device # Save device that x is on.

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = idx_base.long()

    idx = idx + idx_base # (Batch_index, Point_index, TopK_index) -> Global_Point_index

    idx = idx.contiguous().view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous() # (Batch_index, Point_index, Feature_index) -> Feature_value
    # x.view: (Global_point_index, Feature_index) -> Feature_value
    feature = x.view(batch_size*num_points, -1)[idx, :] # (Batch_index, Point_index, TopK_index, Feature_index) -> Feature_value
    feature = feature.view(batch_size, num_points, k[0], num_dims) # Reshape ?Does nothing again?
    # x.view: (Batch_index, Point_index, 1, Feature_index) -> Feature_value
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k[0], 1) # (Batch_index, Point_index, k, Feature_index) -> Feature_value
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2) # (Batch_index, Feature_index, Point_index, (TopK_index+k)) -> Feature_value
  
    return feature

 # x: (batch_index, feature_index, point_index) -> feature_value
 # k: (3 ints named k_g?)
 # idx: (batch_index, point_index, neighbor_index) -> point_index
def get_graph_feature_idx(x, k, idx):
    batch_size = x.size(0) # Get the number of inputs per batch
    num_points = x.size(2) # Get the number of (graph) points (which are faces)
    x = x.view(batch_size, -1, num_points) # Reshape (merges all dimensions aside dimension 0 and 2 into dimension 1 to become the feature dimension)
    # New shape: (batch_index, feature_index, point_index) -> feature_value
    
    device = torch.device('cuda') # Register device cuda

    # Create index base, such that all graphs have individual indices.
    # Example: If input 1 has 64 neighbors, the graph of input 2 starts with index 64, such that none overlap.
    # Shape: (batch_index, 1, 1) -> batch_index*num_points
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = idx_base.long()

    # Actually alter the indices here by adding a base to all indices in idx.
    # inx points from (batch index, point index) to (neighbor index),
    # so only the neihgbor indices are increased in this mapping.
    # (batch_index, point_index, neighbor_index) -> global_point_index
    idx = idx + idx_base

    # Contiguous is a call, which makes a copy from the referenced tensor.
    # This flattens the index mapping of idx
    # New shape: (batch_point_neighbor_index) -> global_point_index
    idx = idx.contiguous().view(-1)
 
    # Get the number of features per graph node.
    _, num_dims, _ = x.size()

    # Transpose the axes 1 and 2 of x.
    # In other words: Swap the feature and (graph)point index axis
    # (Also copy the tensor to a new version in memory)
    # New shape: (Batch_index, point_index, feature_index) -> feature_value
    x = x.transpose(2, 1).contiguous()
    # A newly created feature (probably vector)
    # Batch index and graphpoint index get merged.
    # Reshape x to (global_point_index, feature_index) -> feature_value
    # Also, index using idx:
    # New shape: (batch_point_neighbor_index, feature index) -> feature_value
    feature = x.view(batch_size*num_points, -1)[idx, :]
    # Reshape to (batch_index, point_index, neighbor_index, feature_index) -> feature_value
    feature = feature.view(batch_size, num_points, k[0], num_dims)
    # Reshape -> (batch_index, point_index, 1, feature_index) -> feature_value
    # Repeat -> (batch_index, point_index, neighbor_index, feature_index) -> feature_value
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k[0], 1)
    
    # FROM HERE: x contains the current feature values and feature contains the feature values of the neighbors.

    # Get the difference of features with neighbor feature minus current feature
    # Concatenate the difference and the original feature value in the feature_index_dimension
    # (There are now twice as many features as before. The second half is a copy of the original feature values.)
    # Reorder axes to (Batch_index, feature_index, point_index, neighbor_index)
    # Reorder is done, because 2D convolution is always done over last 2 axes.
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    # Returning feature, which will be used to do 2D convolution over (point_index, neighbor_index)
    return feature

class DGCNN(nn.Module):
    def __init__(self, k, init_dims, emb_dims, dropout, output_channels=3):
        super(DGCNN, self).__init__()
        self.k = torch.IntTensor([k]) # Set to 8 most of the times
        self.k_g = torch.IntTensor([3]) # I don't know what this does. The first integer is indexed and it should give back 3, representing the amount of neighbors per triangle. There is no definition to the second and third value yet.
        
        # Batch normalization layers.
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm1d(emb_dims)

        # Convolutional layers for edge convolution.
        # The input is multiplied by 2, because the difference of features and original feature values are concatenated.
        # You could think of the convolutional layer as constituting a fully connected layer applied at every single pixel location to transform the corresponding input values into output values.
        self.conv1 = nn.Sequential(nn.Conv2d(init_dims*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(256*2, 256, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(256*2, 256, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1024, emb_dims, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(emb_dims*2, 512, bias=False)
        self.bn8 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn9 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, 64)
        self.bn10 = nn.BatchNorm1d(64)
        self.linear4 = nn.Linear(64, output_channels)

    def forward(self, inputs):
        # Features: {Center(3), Normal(3), Area(1), Neighbors(3), Point_positions(9)}
        x = inputs[:, 0:17, :] # (Batch_index, Feature_index, Point_index)
        idx = inputs[:, 17:20, :] # (Batch_index, Neighbor_index, Point_index)
        idx = idx.long() # (Batch_index, Neighbor_index, Point_index)_Long
        idx = idx.permute(0, 2, 1) # (Batch_index, Point_index, Neighbor_index)_Long

        batch_size = x.size(0) # Batch_size (number of batches)
        x = get_graph_feature_idx(x, self.k_g, idx) # (Batch_index, Feature_index(difference with neighbor, original), Point_index, Neighbor_index)
        x = self.conv1(x) # (Batch_index, New_features_index, Point_index, Neighbor_index)
        x1 = x.max(dim=-1, keepdim=False)[0] # (Batch_index, New_features_index, Point_index)

        x = get_graph_feature_idx(x1, self.k_g, idx) # (Batch_index, Feature_index, Point_index, Neighbor_index)
        x = self.conv2(x) # (Batch_index, Feature_index, Point_index, Neighbor_index)
        x2 = x.max(dim=-1, keepdim=False)[0] # (Batch_index, Feature_index, Point_index)

        x = get_graph_feature_idx(x2, self.k_g, idx) # (Batch_index, Feature_index, Point_index, Neighbor_index)
        x = self.conv3(x) # (Batch_index, Feature_index, Point_index, Neighbor_index)
        x3 = x.max(dim=-1, keepdim=False)[0] # (Batch_index, Feature_index, Point_index)

        x = get_graph_feature(x3, k=self.k) # (Batch_index, Feature_index, Point_index, TopK_index) -> Feature_value
        x = self.conv4(x) # (Batch_index, Feature_index, Point_index, TopK_index) -> Feature_value
        x4 = x.max(dim=-1, keepdim=False)[0] # (Batch_index, Feature_index, Point_index) -> Max_Feature_value

        x = get_graph_feature(x4, k=self.k) # (Batch_index, Feature_index, Point_index, TopK_index) -> Feature_value
        x = self.conv5(x) # (Batch_index, Feature_index, Point_index, TopK_index) -> Feature_value
        x5 = x.max(dim=-1, keepdim=False)[0] # (Batch_index, Feature_index, Point_index) -> Max_Feature_value

        x = get_graph_feature(x5, k=self.k) # (Batch_index, Feature_index, Point_index, TopK_index) -> Feature_value
        x = self.conv6(x) # (Batch_index, Feature_index, Point_index, TopK_index) -> Feature_value
        x6 = x.max(dim=-1, keepdim=False)[0] # (Batch_index, Feature_index, Point_index) -> Max_Feature_value

        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1) # (Batch_index, Feature_index(x1, x2, x3, x4, x5, x6), Point_index) -> Max_Feature_value

        x = self.conv7(x) # (Batch_index, Feature_index, Point_index) -> Max_Feature_value
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # (Batch_index, Feature_index, 1) -> (Batch_index, Feature_index) -> Feature_value_max_pooled
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1) # (Batch_index, Feature_index, 1) -> (Batch_index, Feature_index) -> Feature_value_avg_pooled
        x = torch.cat((x1, x2), 1) # (Batch_index, Feature_index(x1+x2)) -> Feature_value_pooled

        x = F.leaky_relu(self.bn8(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn9(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = F.leaky_relu(self.bn10(self.linear3(x)), negative_slope=0.2)
        x = self.linear4(x)
        return x

class BetterDGCNN(nn.Module):
    def __init__(self, l_e, l_d, l_l, channel_sizes, k, init_dims, dropout, output_channels=3):
        super(BetterDGCNN, self).__init__()
        self.l_e = l_e
        self.l_d = l_d
        self.l_l = l_l
        self.channel_sizes = channel_sizes

        self.k = torch.IntTensor([k])
        self.k_g = torch.IntTensor([3])
        self.cum_sizes = F.pad(torch.cumsum(channel_sizes, 0), (1, 0), "constant", value=0)
        
        for i in range(l_e+l_d+1):
            bn_attr = f"bn{(i+1)}"
            bn_val = (nn.BatchNorm1d if i == l_e + l_d else nn.BatchNorm2d)(channel_sizes[i])
            conv_attr = f"conv{(i+1)}"
            conv_input = init_dims*2 if i == 0 else (self.cum_sizes[i] if i == l_e+l_d else channel_sizes[i-1]*2)
            conv_output = channel_sizes[i]
            conv_val = nn.Sequential((nn.Conv1d if i == l_e + l_d else nn.Conv2d)(conv_input, conv_output, kernel_size=1, bias=False),
                                     bn_val,
                                     nn.LeakyReLU(negative_slope=0.2))
            setattr(self, bn_attr, bn_val)
            setattr(self, conv_attr, conv_val)
        
        for i in range(l_l-1):
            linear_attr = f"linear{(i+1)}"
            in_features = channel_sizes[i + l_e + l_d]*(2 if i==0 else 1)
            out_features = channel_sizes[i + l_e + l_d + 1]
            linear_val = nn.Linear(in_features, out_features, bias=i>0)
            bn_attr = f"bn{(i+l_e+l_d+2)}"
            bn_val = nn.BatchNorm1d(channel_sizes[i + l_e + l_d + 1])
            if i < l_l - 2:
                dp_attr = f"dp{(i+1)}"
                dp_val = nn.Dropout(p=dropout)
                setattr(self, dp_attr, dp_val)
            setattr(self, linear_attr, linear_val)
            setattr(self, bn_attr, bn_val)
        final_linear_attr = f"linear{l_l}"
        final_linear_val = nn.Linear(channel_sizes[-1], output_channels)
        setattr(self, final_linear_attr, final_linear_val)

    def forward(self, inputs):
        device = inputs.device
        x = inputs[:, 0:17, :]
        idx = inputs[:, 17:20, :]
        idx = idx.long()
        idx = idx.permute(0, 2, 1)
        batch_size = x.size(0)
        num_points = x.size(2)

        concatted_tensor = torch.empty(batch_size, sum(self.channel_sizes[:(self.l_d+self.l_e)]), num_points, device=device)

        # Edge Convolution Layers
        for i in range(self.l_e):
            x = get_graph_feature_idx(x, self.k_g, idx)
            x = getattr(self, f"conv{i+1}")(x)
            x = x.max(dim=-1, keepdim=False)[0]
            concatted_tensor[:, self.cum_sizes[i]:self.cum_sizes[i+1], :] = x
        
        # Dynamic Edge Convolution Layers
        for i in range(self.l_d):
            x = get_graph_feature(x, k=self.k)
            x = getattr(self, f"conv{self.l_e+i+1}")(x)
            x = x.max(dim=-1, keepdim=False)[0]
            concatted_tensor[:, self.cum_sizes[self.l_e + i]:self.cum_sizes[self.l_e + i+1], :] = x

        # Concatenation, Linear Layer and Pooling
        x = getattr(self, f"conv{self.l_e+self.l_d+1}")(concatted_tensor)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        # Fully Connected Regression layers
        for i in range(self.l_l-1):
            x = F.leaky_relu(getattr(self, f"bn{self.l_e+self.l_d+2+i}")(getattr(self, f"linear{i+1}")(x)))
            if i < self.l_l - 2:
                x = getattr(self, f"dp{i+1}")(x)
        
        # Output layer
        x = getattr(self, f"linear{self.l_l}")(x)
        return x

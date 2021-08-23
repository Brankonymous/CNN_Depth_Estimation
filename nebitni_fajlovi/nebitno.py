def get_batch(self, batch_size=32):
    # Array of indices
    ind = np.arange(self.N)
    # Indices of a batch
    ind_batch = sample(ind.tolist(), batch_size)
    # Transform to tensor
    tensor_images = torch.from_numpy(self.images[ind_batch,:,:,:]).float()
    tensor_depths = torch.from_numpy(self.depths[ind_batch,:,:,:]).float()

    return tensor_images, tensor_depths
if epoch >= 1:
    bbox_list = []
    outputs = outputs.detach().cpu().numpy()
    # Process the batch outputs
    for output in outputs:
        # Apply argmax and thresholding
        output = np.argmax(output, axis=0)
        output[output >= 1] = 1

        # Label connected components
        labeled_array, num_features = label(output, structure=[[1, 1, 1],
                                                               [1, 1, 1],
                                                               [1, 1, 1]])

        # Get unique labels and their counts
        labels, counts = np.unique(labeled_array, return_counts=True)

        # Remove background label (0)
        labels = labels[1:]
        counts = counts[1:]

        largest_patch_index = labels[np.argmax(counts)]
        largest_patch_indices = np.where(labeled_array == largest_patch_index)

        bbox = find_objects(labeled_array == largest_patch_index)[0]
        bbox_list.append(bbox)
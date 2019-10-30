
#resnet_weigth_path = 'data/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
#fine_tuned_resnet_weight_path = 'ResNet50/logs/ft-41-0.87.hdf5'
#learning_model.add(ResNet50(include_top = False, pooling='ave', weights = resnet_weigth_path))
#learning_model.layers[0].trainable = False
'''
elif weight_type == 'ft':
    num_classes = 9
    base_model = ResNet50
    base_model = base_model(weights = 'imagenet', include_top = False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation= 'relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    clustering_model = Model(inputs = base_model.input, outputs = predictions)
    clustering_model.load_weights(fine_tuned_resnet_weight_path)

    clustering_model.layers.pop()
    clustering_model.layers.pop()
    clustering_model.outputs = [clustering_model.layers[-1].output]
    clustering_model.layers[-1].outbound_nodes = []

    clustering_model.layers[0].trainable = False
'''

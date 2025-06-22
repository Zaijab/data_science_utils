def test_means_plot() -> None:
    key = jax.random.key(0)
    keys = jax.random.split(key, 10)

    # Number of points for each cluster
    pts = 50
    # Mean vector for first cluster
    mu_a = jnp.array([0, 0])
    # Covariance matrix for first cluster
    cov_a = jnp.array([[4, 1], [1, 4]])
    # Sampled points for the first cluster
    a = random.multivariate_normal(keys[0], mu_a, cov_a, shape=(pts,))
    # Mean vector for second cluster
    mu_b = jnp.array([30, 10])
    # Covariance matrix for second cluster
    cov_b = jnp.array([[10, 2], [2, 1]])
    # Sampled points for the second cluster
    b = random.multivariate_normal(keys[1], mu_b, cov_b, shape=(pts,))
    # combined points
    features = jnp.concatenate((a, b))
    # plot the points
    plt.scatter(features[:, 0], features[:, 1])

    # number of clusters
    k=2
    # Perform K-means clustering
    result = kmeans(keys[3], features, k)
    centroids = result.centroids
    assignment = result.assignment
    for i in range(k):
        # points for the k-th cluster
        cluster = features[assignment == i]
        plt.plot(cluster[:,0], cluster[:,1], "o", alpha=0.4)
    # plot the centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], c='r')

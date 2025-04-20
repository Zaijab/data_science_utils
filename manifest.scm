(use-modules (gnu packages)
	     (guix download)
	     (guix profiles)
	     (guix build-system python)
	     (guix packages)
	     (guix git-download)
	     (guix build-system pyproject)
	     ((guix licenses) #:prefix license:)
	     (gnu packages python-build)
	     (gnu packages image-processing)
	     (gnu packages python)
	     (gnu packages jupyter)
	     (gnu packages check)
	     (gnu packages version-control)
	     (gnu packages python-xyz)
	     (gnu packages python-check)
	     (gnu packages pdf)
	     (gnu packages python-science)
	     (gnu packages machine-learning)
	     (guix-science packages python)
	     (guix-science packages machine-learning))

(define-public python-jaxtyping-three
  (package
    (name "python-jaxtyping")
    (version "0.3.0")
    (source (origin
              (method url-fetch)
              (uri (pypi-uri "jaxtyping" version))
              (sha256
               (base32
                "1pk8m47b7cy6fyl7yp63r403fka090alav0bvnnk4lr96rjbad5k"))))
    (build-system pyproject-build-system)
    ;; Tests require JAX, but JAX can't be packaged because it uses the Bazel
    ;; build system.
    (arguments (list #:tests? #f))
    (native-inputs (list python-hatchling))
    (propagated-inputs (list python-numpy python-wadler-lindig python-typeguard
                             python-typing-extensions))
    (home-page "https://github.com/google/jaxtyping")
    (synopsis
     "Type annotations and runtime checking for JAX arrays and others")
    (description "@code{jaxtyping} provides type annotations and runtime
checking for shape and dtype of JAX arrays, PyTorch, NumPy, TensorFlow, and
PyTrees.")
    (license license:expat)))

(define-public python-flax
  (package
    (name "python-flax")
    (version "0.10.0")
    (source
     (origin
       (method git-fetch)
       (uri (git-reference
             (url "https://github.com/google/flax")
             (commit (string-append "v" version))))
       (file-name (git-file-name name version))
       (sha256
        (base32 "195m9wdjdp2s47mzd1nl5cvkf7xhiaq87hddwa8kfx9gd9fzcgjy"))))
    (build-system pyproject-build-system)
    (arguments
     (list
      #:test-flags
      '(list "--pyargs" "tests"
             ;; We don't have tensorboard
             "--ignore=tests/tensorboard_test.py"
             ;; These tests are failing bacause flax might only work
             ;; on CPUs that have AVX support.
             "--ignore=tests/serialization_test.py"
             "--ignore=tests/linen/linen_test.py"
             "--ignore=tests/linen/linen_recurrent_test.py"
             "--ignore=tests/linen/linen_dtypes_test.py"
             ;; These tests try to use a fixed number of CPUs that may
             ;; exceed the number of CPUs available at build time.
             "--ignore=tests/jax_utils_test.py"
	     "--ignore=tests/flaxlib_test.py"
	     "--ignore=tests/nnx/transforms_test.py")
      #:phases
      '(modify-phases %standard-phases
         (add-after 'unpack 'ignore-deprecations
           (lambda _
             (substitute* "pyproject.toml"
               (("\"error\",") "")))))))
    (propagated-inputs
     (list python-einops
           python-jax
           python-optax
           python-orbax-checkpoint
           python-msgpack
           python-numpy
           python-pyyaml
           python-rich
           python-tensorstore
           python-typing-extensions))
    (native-inputs
     (list opencv
           python-nbstripout
           python-ml-collections
           python-mypy
           python-pytorch
           python-pytest
           python-pytest-cov
           python-pytest-xdist
           python-setuptools-scm
           python-tensorflow))
    (home-page "https://github.com/google/flax")
    (synopsis "Neural network library for JAX designed for flexibility")
    (description "Flax is a neural network library for JAX that is
designed for flexibility.")
    (license license:asl2.0)))

(define-public python-equinox
  (package
    (name "python-equinox")
    (version "0.11.10")
    (source
     (origin
       (method git-fetch)
       (uri (git-reference
             (url "https://github.com/patrick-kidger/equinox")
             (commit (string-append "v" version))))
       (file-name (git-file-name name version))
       (sha256
        (base32 "1gm8xxx1inbzjfqvd6224k3arbcf7dyfk258rbbkl22nvcnv12j2"))))
    (build-system pyproject-build-system)
    (arguments (list #:tests? #f))
    (native-inputs (list python-hatchling))
    (propagated-inputs (list python-jax
			     python-jaxtyping-three
                             python-typing-extensions))
    (home-page "https://github.com/google/jaxtyping")
    (synopsis
     "Type annotations and runtime checking for JAX arrays and others")
    (description "@code{jaxtyping} provides type annotations and runtime
checking for shape and dtype of JAX arrays, PyTorch, NumPy, TensorFlow, and
PyTrees.")
    (license license:expat)))

(define-public python-optax
  (package
    (name "python-optax")
    ;; 0.1.6 needs a more recent numpy
    (version "0.2.4")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "optax" version))
       (sha256
        (base32 "0dkkyh2a6j94k6g5xljfbl1hqfndwqvax1wi656dwvby63ax61af"))))
    (build-system pyproject-build-system)
    ;; Tests require haiku, tensorflow, and flax, but flax needs
    ;; optax.
    (arguments
     (list #:tests? #false))
    (propagated-inputs (list python-absl-py
                             python-chex
                             python-jax
                             python-jaxlib
                             python-numpy))
    (native-inputs
     (list python-dm-tree
           python-pytest
           python-setuptools	   
	   python-flit-core
	   python-etils
           python-wheel))
    (home-page "https://github.com/google-deepmind/optax/")
    (synopsis "Gradient processing and optimization library for JAX")
    (description "Optax is a gradient processing and optimization
library for JAX.")
    (license license:asl2.0)))


(define-public python-lineax
  (package
    (name "python-lineax")
    (version "0.0.7")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "lineax" version))
       (sha256
        (base32 "0j7a80lgg059nan7nj33ng6180hsfik4irdgd132shq2sal4jdg4"))))
    (build-system pyproject-build-system)
    (arguments (list #:tests? #f))
    (propagated-inputs (list python-equinox python-jax python-jaxtyping-three
                             python-typing-extensions python-typing-extensions))
    (native-inputs (list python-hatchling))
    (home-page #f)
    (synopsis "Linear solvers in JAX and Equinox.")
    (description "Linear solvers in JAX and Equinox.")
    (license #f)))

(define-public python-ott-jax
  (package
    (name "python-ott-jax")
    (version "0.5.0")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "ott_jax" version))
       (sha256
        (base32 "0ljyfvq7ylsyhsxkj5gygsbzc7x7p0d28avawq323inlflba9bha"))))
    (build-system pyproject-build-system)
    (arguments (list #:tests? #f))
    (native-inputs (list python-hatchling python-setuptools python-etils))
    (propagated-inputs (list python-jax
                             python-jaxopt
                             python-lineax
                             python-numpy
                             python-optax
                             python-typing-extensions))
    (home-page #f)
    (synopsis "Optimal Transport Tools in JAX")
    (description "Optimal Transport Tools in JAX.")
    (license #f)))



(packages->manifest (list
		     python
		     jupyter
		     python-ott-jax
		     python-numpy
		     python-pandas
		     python-matplotlib
		     python-seaborn
		     python-scikit-learn
		     python-pytorch
		     python-tensorflow
		     python-jax
		     python-jaxtyping-three
		     python-beartype
		     python-equinox
		     python-flax))

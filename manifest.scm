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
	     (guix-science packages machine-learning)
	     (guix-science-nonfree packages machine-learning)
	     )

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


(define-public python-evosax
  (package
    (name "python-evosax")
    (version "0.1.6")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "evosax" version))
       (sha256
        (base32 "0js855gjdp0lkd87kcw0bv4qzl3hsa40qzhj125qfbqz34rd0gdw"))))
    (build-system python-build-system)
    (arguments (list #:tests? #f))
    (native-inputs (list python-hatchling python-setuptools python-etils))
    (propagated-inputs (list python-jax
                             python-jaxlib
                             python-chex
                             python-flax
                             python-numpy
                             python-pyyaml
			     python-matplotlib
			     python-dotmap))
    (home-page #f)
    (synopsis "Optimal Transport Tools in JAX")
    (description "Optimal Transport Tools in JAX.")
    (license #f)))

(define-public python-evosax-2
  (package
    (name "python-evosax")
    (version "0.2.0")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "evosax" version))
       (sha256
        (base32 "1mmx9vxfjc4hxjv0jrh9lr23cwi2sws3lixx8s1qmwd37szckvvz"))))
    (build-system pyproject-build-system)
    (arguments
      `(#:tests? #f
        #:phases
         (modify-phases %standard-phases
           ;; Package is not loadable on its own at this stage.
           (delete 'sanity-check))))
    (native-inputs (list python-hatchling python-setuptools python-etils))
    (propagated-inputs (list python-jax
                             python-jaxlib
                             python-chex
                             python-flax
                             python-numpy
                             python-pyyaml
			     python-matplotlib
			     python-dotmap))
    (home-page #f)
    (synopsis "Optimal Transport Tools in JAX")
    (description "Optimal Transport Tools in JAX.")
    (license #f)))

(define-public python-dm-env
  (package
    (name "python-dm-env")
    (version "1.6")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "dm-env" version))
       (sha256
        (base32 "1plk7pzzrd3yz5izzkhga45i99xy33icw5m5hv4y0facclffndm4"))))
    (build-system pyproject-build-system)
    (native-inputs (list python-setuptools))
    (propagated-inputs (list python-absl-py
			     python-dm-tree
			     python-numpy
			     python-pytest))
    (home-page #f)
    (synopsis "")
    (description "")
    (license #f)))


(define-public python-waymax
  (package
    (name "python-waymax")
    (version "720f9214a9bf79b3da7926497f0cd0468ca3e630")
    (source
     (origin
       (method git-fetch)
       (uri (git-reference
             (url "https://github.com/waymo-research/waymax.git")
             (commit version)))
       (file-name (git-file-name name version))
       (sha256
        (base32 "0j6qg49rixqn0cpphf31hfjd1kap61n2pfm1lyjlfv0kq3j6jm07"))))
    (build-system pyproject-build-system)
    #;(arguments
      `(#:tests? #f
        #:phases
         (modify-phases %standard-phases
           ;; Package is not loadable on its own at this stage.
           (delete 'sanity-check))))
    (native-inputs (list python-setuptools))
    (propagated-inputs (list
			python-numpy
			python-jax
			python-tensorflow
			python-chex
			python-dm-env
			python-flax
			python-matplotlib
			python-dm-tree
			python-immutabledict
			python-pillow
			python-mediapy
			python-tqdm
			python-absl-py))
    (home-page #f)
    (synopsis "")
    (description "")
    (license #f)))

(define-public python-easydict
  (package
    (name "python-easydict")
    (version "1.13")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "easydict" version))
       (sha256
        (base32 "103pr3b4j53r2rsci8g1pi1alzscfk4pxxy15c703j21pknms4xi"))))
    (build-system pyproject-build-system)
    (native-inputs (list python-setuptools python-wheel))
    (home-page "https://github.com/makinacorpus/easydict")
    (synopsis "Access dict values as attributes (works recursively).")
    (description "Access dict values as attributes (works recursively).")
    (license #f)))

(define-public python-motmetrics
  (package
    (name "python-motmetrics")
    (version "v1.4.0")
    (source
     (origin
       (method git-fetch)
       (uri (git-reference
             (url "https://github.com/cheind/py-motmetrics.git")
             (commit version)))
       (file-name (git-file-name name version))
       (sha256
        (base32 "000rmpx373la1rd4mb5886sx7ap12j7jrwi6468skkr63zdqlih5"))))
    (build-system python-build-system)
    (arguments '(#:tests? #false))
    (propagated-inputs (list
	     python-lap
	     python-setuptools
	     python-pytest
	     python-numpy
	     python-pandas
	     python-scipy
	     python-xmltodict
	     ))
    (native-inputs (list python-setuptools
			 python-pytest
			 python-numpy
			 python-pandas
			 python-scipy
			 python-xmltodict))
    (home-page "https://github.com/makinacorpus/easydict")
    (synopsis "Access dict values as attributes (works recursively).")
    (description "Access dict values as attributes (works recursively).")
    (license #f)))

(define-public python-pyquaternion
  (package
    (name "python-pyquaternion")
    (version "v0.9.9")
    (source
     (origin
       (method git-fetch)
       (uri (git-reference
             (url "https://github.com/KieranWynn/pyquaternion.git")
             (commit version)))
       (file-name (git-file-name name version))
       (sha256
        (base32 "0w7vr4ddp4fycrkj3vnc19cm1yrn0diadknyk0k1f3al67s16k1g"))))
    (build-system pyproject-build-system)
    (propagated-inputs (list python-numpy))
    (native-inputs (list python-mkdocs python-nose python-setuptools
                         python-wheel))
    (home-page "http://kieranwynn.github.io/pyquaternion/")
    (synopsis
     "A fully featured, pythonic library for representing and using quaternions.")
    (description
     "This package provides a fully featured, pythonic library for representing and
using quaternions.")
    (license license:expat)))

(define-public python-shapely-for-python-descartes
  (package
    (name "python-shapely-for-descartes")
    (version "1.7.0")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "shapely" version))
       (sha256
        (base32 "0cpyziixzdj7xqkya4k6fwr0qmrw8k84fsrx6p5sdgw6qxmkdwmz"))))
    (build-system pyproject-build-system)
    (arguments
     (list
      #:phases
      '(modify-phases %standard-phases
         (add-before 'check 'build-extensions
           (lambda _
             ;; Cython extensions have to be built before running the tests.
             (invoke "python" "setup.py" "build_ext" "--inplace"))))))
    (native-inputs
     (list python-cython python-matplotlib python-pytest python-setuptools python-wheel))
    (inputs
     (list geos))
    (propagated-inputs
     (list python-numpy))
    (home-page "https://github.com/shapely/shapely")
    (synopsis "Library for the manipulation and analysis of geometric objects")
    (description "Shapely is a Python package for manipulation and analysis of
planar geometric objects.  It is based on the @code{GEOS} library.")
    (license license:bsd-3)))

(define-public python-descartes
  (package
    (name "python-descartes")
    (version "0842f2d21c4fff7f4a3fd91a194de6408842720d")
    (source
     (origin
       (method git-fetch)
       (uri (git-reference
             (url "https://github.com/benjimin/descartes.git")
             (commit version)))
       (file-name (git-file-name name version))
       (sha256
        (base32 "11vqpa3y41167vx1v0kc3p44b9x7dv6z80qj1m0yq759ljzb9kq4"))))
    (build-system pyproject-build-system)
    (arguments (list #:phases
		     '(modify-phases %standard-phases
			(delete 'check))))
    (propagated-inputs (list python-numpy python-matplotlib))
    (native-inputs (list python-setuptools python-wheel))
    (home-page "http://bitbucket.org/sgillies/descartes/")
    (synopsis "Use geometric objects as matplotlib paths and patches")
    (description "Use geometric objects as matplotlib paths and patches.")
    (license license:bsd-3)))

(define-public python-pycocotools
  (package
    (name "python-pycocotools")
    (version "2.0.8")
    (source
     (origin
       (method url-fetch)
       (uri (pypi-uri "pycocotools" version))
       (sha256
        (base32 "13gv3lg042jy43myld1brgfdjajvxg2gk01ng8v6r8kbg3dwwawg"))))
    (build-system pyproject-build-system)
    (propagated-inputs (list python-matplotlib python-numpy))
    (native-inputs (list python-cython python-numpy python-setuptools
                         python-wheel))
    (home-page "https://github.com/ppwwyyxx/cocoapi")
    (synopsis "Official APIs for the MS-COCO dataset")
    (description "Official APIs for the MS-COCO dataset.")
    (license #f)))

(define-public python-nuscenes-devkit
  (package
    (name "python-nuscenes-devkit")
    (version "932064c611ddfe06e3d1fadea904eb365482a03b")
    (source
     (origin
       (method git-fetch)
       (uri (git-reference
             (url "https://github.com/nutonomy/nuscenes-devkit.git")
             (commit version)))
       (file-name (git-file-name name version))
       (sha256
        (base32 "12mhvszkckfvl75lpr5p5vym7hlgklj1sdx19vihc75da9h82p4y"))))
    (build-system python-build-system)
    (arguments
     (list
      #:tests? #f   ;; Disable tests as they require large external data
      #:phases
      #~(modify-phases %standard-phases
          ;; Create a custom setup.py in the root directory based on SO solution
          (add-before 'build 'create-root-setup-py
            (lambda _
              (with-output-to-file "setup.py"
                (lambda ()
                  (display "
from pathlib import Path
import setuptools

package_dir = 'python-sdk'
setup_dir = Path(__file__).parent / 'setup'

# Since nuScenes 2.0 the requirements are stored in separate files.
requirements = []
for req_path in (setup_dir / 'requirements.txt').read_text().splitlines():
    if req_path.startswith('#'):
        continue
    if req_path.startswith('-r '):
        req_path = req_path.replace('-r ', '')
        requirements += (setup_dir / req_path).read_text().splitlines()
    else:
        requirements.append(req_path)

setuptools.setup(
    name='nuscenes-devkit',
    version='1.1.9',
    author='Holger Caesar, Oscar Beijbom, Qiang Xu, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, '
           'Sergi Widjaja, Kiwoo Shin, Caglayan Dicle, Freddy Boulton, Whye Kit Fong, Asha Asvathaman, Lubing Zhou '
           'et al.',
    author_email='nuscenes@motional.com',
    description='The official devkit of the nuScenes dataset (www.nuscenes.org).',
    long_description=Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    url='https://github.com/nutonomy/nuscenes-devkit',
    python_requires='>=3.6',
    install_requires=requirements,
    packages=setuptools.find_packages(package_dir),
    package_dir={'': package_dir},
    package_data={'': ['*.json']},
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Operating System :: OS Independent',
        'License :: Free for non-commercial use'
    ],
    license='cc-by-nc-sa-4.0'
)
")))
              #t))
              
          ;; Skip original build and use our custom setup.py
          (replace 'build
            (lambda _
              (invoke "python" "setup.py" "build")))
          
          ;; Fix the install phase to correctly set the prefix
          (replace 'install
            (lambda* (#:key outputs #:allow-other-keys)
              (let ((out (assoc-ref outputs "out")))
                (invoke "python" "setup.py" "install"
                        "--prefix" out
                        "--single-version-externally-managed"
                        "--root" "/"))))
          
          ;; Add a phase to create a direct-install alternative if the proper way fails
          (add-after 'install 'direct-install
            (lambda* (#:key outputs #:allow-other-keys)
              (let* ((out (assoc-ref outputs "out"))
                     (site-packages (string-append out "/lib/python"
                                                  #$(version-major+minor (package-version python))
                                                  "/site-packages"))
                     (target-path (string-append site-packages "/nuscenes")))
                ;; Create target directories if they don't exist
                (mkdir-p site-packages)
                
                ;; Copy the core nuscenes module
                (when (not (file-exists? target-path))
                  (copy-recursively "python-sdk/nuscenes" target-path))
                
                ;; Create a simple __init__.py if needed
                (with-output-to-file (string-append site-packages "/nuscenes/__init__.py")
                  (lambda () 
                    (display "")))
                
                ;; Create a .pth file to ensure the package is in the path
                (with-output-to-file (string-append site-packages "/nuscenes.pth")
                  (lambda ()
                    (display (string-append site-packages "/nuscenes\n"))))
                #t)))
          
          ;; Add a phase to check what was installed
          (add-after 'direct-install 'check-installation
            (lambda* (#:key outputs #:allow-other-keys)
              (let ((out (assoc-ref outputs "out")))
                (format #t "~%Installation directory contents:~%")
                (system (string-append "find " out " -type f -name '*.py' | grep -v '__pycache__' | sort | head -20"))
                #t)))
	  
	  (delete 'sanity-check)
	  
	  )))
    
    ;; Declare runtime dependencies
    (propagated-inputs
     (list python-setuptools
           python-cachetools
           python-descartes
           python-fire
           python-matplotlib
           python-numpy
           python-pillow
           python-pyquaternion
           python-scikit-learn
           python-scipy
	   python-shapely-for-python-descartes
           python-tqdm
           python-parameterized
           python-pycocotools
           python-pytest
           opencv))
    (home-page "https://github.com/nutonomy/nuscenes-devkit")
    (synopsis "Python devkit for the nuScenes dataset")
    (description
     "Official Python devkit for the nuScenes autonomous driving dataset.")
    (license #f)))  ;; Replace with appropriate license if known

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
		     ;; python-jax-with-cuda11
		     python-jaxtyping-three
		     python-beartype
		     python-equinox
		     python-flax
		     
		     python-waymax
		     python-evosax))



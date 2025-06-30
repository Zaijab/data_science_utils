(use-modules (guix ci)
	     (ice-9 popen)
	     (guix channels))

(list
 (channel
  (name 'guix-science)
  (url "https://codeberg.org/guix-science/guix-science.git")
  (branch "master")
  (commit
   "bc3f3d3863c31969dc195d420a356f357db93ac7")
  (introduction
   (make-channel-introduction
    "b1fe5aaff3ab48e798a4cce02f0212bc91f423dc"
    (openpgp-fingerprint
     "CA4F 8CF4 37D7 478F DA05  5FD4 4213 7701 1A37 8446"))))
 (channel
  (name 'guix-science-nonfree)
  (url "https://codeberg.org/guix-science/guix-science-nonfree.git")
  (branch "master")
  (commit
   "18bc80bd6e640c4f89a48efbc4c153a886a7aa87")
  (introduction
   (make-channel-introduction
    "58661b110325fd5d9b40e6f0177cc486a615817e"
    (openpgp-fingerprint
     "CA4F 8CF4 37D7 478F DA05  5FD4 4213 7701 1A37 8446"))))
 (channel
  (name 'guix)
  (url "https://git.savannah.gnu.org/git/guix.git")
  (branch "master")
  (commit
   "00291ad00ffddde1e557defaec3e9fde3f20cfaf")
  (introduction
   (make-channel-introduction
    "9edb3f66fd807b096b48283debdcddccfea34bad"
    (openpgp-fingerprint
     "BBB0 2DDF 2CEA F6A8 0D1D  E643 A2A0 6DF2 A33A 54FA")))))


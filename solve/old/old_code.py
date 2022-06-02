# def make_art_gratings(img1, img2):
#     # dim = (max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1]) )
#     #
#     # img1 = np.pad(img1, ((0, max(0, dim[0] - img1.shape[0])), (0, max(0, dim[1] - img1.shape[1]))))
#     # img2 = np.pad(img2, ((0, max(0, dim[0] - img2.shape[0])), (0, max(0, dim[1] - img2.shape[1]))))
#
#     img1 = resize(img1, (512,512))
#     img2 = resize(img2 , (512, 512))
#     img1 = (img1 - np.min(img1))/(np.max(img1) - np.min(img1))
#     img2 = (img2 - np.min(img2))/(np.max(img2) - np.min(img2))
#
#     img1 = prep_img(img1)
#     img2 = prep_img(img2)
#
#
#     plt.imshow(img1, cmap='gray')
#     plt.show()
#     plt.imshow(img2, cmap='gray')
#     plt.show()
#
#     shift = int(0.2 * 512)
#     xcors, ycors  = np.meshgrid(np.arange(512), np.arange(512))
#     xcors_extra, ycors_extra  = np.meshgrid(np.arange(512), np.arange(512  + shift))
#
#     p1 = np.vectorize(lambda x: 0.5 + 0.5*np.cos(x))
#     p2 = p1
#
#
#
#     phase_mod1 = (np.arccos(-2+8*img1))/(2*np.pi)
#     phase_mod1 *= 1/np.mean(img1)
#
#     phase_mod2 = (np.arccos(-2+8*img2))/(2*np.pi)
#     phase_mod2 *= 1/np.mean(img2)
#
#     x = minimize(energy, [0.5*np.pi, 0, 0.5*np.pi, 0], (phase_mod1, phase_mod2, shift),
#                  options={'maxiter':11})
#     print(x)
#     a1 = (x.x)[0]
#     b1 = (x.x)[1]
#     a2 = (x.x)[2]
#     b2 = (x.x)[3]
#
#     phi_1 = np.vectorize(lambda x,y: a1 *x + b1*y)
#     phi_2 = np.vectorize(lambda x,y: a2*x + b2*y)
#     L2 = p2(phi_2(xcors_extra,ycors_extra))
#     L1 = p1(phi_1(xcors,ycors))
#
#     plt.imshow(L1, cmap='gray')
#     plt.show()
#     plt.imshow(L2, cmap='gray')
#     plt.show()
#
#     sup1 = L2 * np.pad(L1, ((0, shift), (0,0)), constant_values=1)
#     sup2 = L2 * np.pad(L1, ((shift,0), (0,0)), constant_values=1 )
#
#     plt.imshow(sup1, cmap='gray')
#     plt.show()
#     plt.imshow(sup2, cmap='gray')
#     plt.show()
#
#
#
#
#
# def energy(x, phase_mod1, phase_mod2, shift):
#     a1 = x[0]
#     b1 = x[1]
#     a2 = x[2]
#     b2 = x[3]
#
#     phi_1 = np.vectorize(lambda x,y: a1*x + b1*y)
#     phi_2 = np.vectorize(lambda x,y: a2*x + b2*y)
#
#     xcors, ycors  = np.meshgrid(np.arange(512), np.arange(512))
#     phase_img_1 = phi_1(xcors, ycors)
#     phase_img_2 = phi_2(xcors, ycors)
#
#     unique_constraint = (np.sum(np.abs(phase_img_1[:3, :])**2)
#                         + np.sum(np.abs(phase_img_2[:3, :]**2)))
#
#     obj = (np.sum(np.abs(phase_img_1 - phase_img_2 - phase_mod1)**2)
#             + np.sum(np.abs(phase_img_1 - phi_2(xcors, ycors+shift) - phase_mod2)**2))
#
#     print(f"energy: {obj + unique_constraint}")
#     return obj + unique_constraint

# import cv2
# import numpy as np
# import os


# def save_heat_colorbar_real_fake(
#     save_path="patch_score_colorbar.png",
#     height=256,
#     width=50,
#     cmap=cv2.COLORMAP_JET,
# ):
#     """
#     Saves a vertical heatmap colorbar with 'Real' (low) and 'Fake' (high) labels
#     """

#     # -----------------------------
#     # Create vertical gradient
#     # -----------------------------
#     gradient = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)

#     # Apply colormap
#     colorbar = cv2.applyColorMap(gradient, cmap)
#     colorbar = cv2.resize(
#         colorbar, (width, height), interpolation=cv2.INTER_NEAREST
#     )

#     # Convert BGR → RGB
#     colorbar = cv2.cvtColor(colorbar, cv2.COLOR_BGR2RGB)

#     # -----------------------------
#     # Add text labels
#     # -----------------------------
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1.0
#     thickness = 2
#     color = (255,255,255)  # black

#     # "Fake" (top = high score)
#     cv2.putText(
#         colorbar,
#         "R",
#         (14, 30),
#         font,
#         font_scale,
#         color,
#         thickness,
#         cv2.LINE_AA,
#     )

#     # "Real" (bottom = low score)
#     cv2.putText(
#         colorbar,
#         "F",
#         (14, height - 10),
#         font,
#         font_scale,
#         color,
#         thickness,
#         cv2.LINE_AA,
#     )

#     # -----------------------------
#     # Save
#     # -----------------------------
#     os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
#     cv2.imwrite(save_path, colorbar)

#     print(f"Saved colorbar with labels to: {save_path}")
    
# save_heat_colorbar_real_fake(
#     save_path=os.path.join("patch_score_colorbar.png")
# )



import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_prob_distribution(pred_prob, labels, bins=30,data_name='DFDCP',exp_name="experiment"):
    """
    pred_prob : np.ndarray, shape (N,) – predicted probability for class=1 (fake)
    labels    : np.ndarray, shape (N,) – 0 = real, 1 = fake
    """

    pred_prob = np.asarray(pred_prob)
    labels = np.asarray(labels)

    real_probs = pred_prob[labels == 0]
    fake_probs = pred_prob[labels == 1]

    plt.figure(figsize=(7, 5))
    plt.hist(
        real_probs,
        bins=bins,
        density=True,
        alpha=0.6,
        label="Real"
    )
    plt.hist(
        fake_probs,
        bins=bins,
        density=True,
        alpha=0.6,
        label="Fake"
    )

    plt.xlabel("Predicted Probability (Fake)")
    plt.ylabel("Density")
    plt.title("Prediction Probability Distribution_"+data_name)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    os.makedirs("visualizations/pdfs/"+exp_name, exist_ok=True)
    plt.savefig("visualizations/pdfs/"+exp_name+"/probability_distribution_"+data_name+".png")
# ------------------ Example usage ------------------
exp_name="my_output_clip_L_PatchClassifierCNN3D_use_mean_pred"
pred_label_name="visualizations/saved_patch_scores/"+exp_name

data_name=["DFDCP","Celeb-DF-v1","DFDC"] 
for data_name in data_name:
    pred_label_path=pred_label_name+"/"+data_name+".pkl"
    pred_label=pickle.load(open(pred_label_path, 'rb'))
    predictions=pred_label['predictions']
    labels=pred_label['labels']



    plot_prob_distribution(predictions, labels, data_name=data_name, exp_name=exp_name)
    
    
    
# predictions = [0.43309787, 0.99999917, 0.4329708 , 0.03498263, 0.9704823 ,
#        0.562638  , 0.14964621, 0.76625353, 0.103761  , 0.14807013,
#        0.02237075, 0.01035583, 0.3594911 , 0.26364636, 0.00408588,
#        0.54419994, 0.04916604, 0.13354778, 0.323432  , 0.3683653 ,
#        0.10891176, 0.09509808, 0.7331384 , 0.35401618, 0.01202718,
#        0.06599756, 0.2542647 , 0.39965123, 0.91118264, 0.76019603,
#        0.01210121, 0.9999354 , 0.9997296 , 0.9897697 , 0.04629571,
#        0.3140324 , 0.11364521, 0.7755259 , 0.03559056, 0.3574325 ,
#        0.99406093, 0.9659018 , 0.40633315, 0.20521633, 0.0177005 ,
#        0.07102302, 0.46657866, 0.9310194 , 0.99999356, 0.07122727,
#        0.08701076, 0.06667823, 0.01802531, 0.9075669 , 0.12857008,
#        0.99999166, 0.08012208, 0.11116387, 0.99333376, 0.9923058 ,
#        0.8952642 , 0.3336965 , 0.18556267, 0.7426398 , 0.6147175 ,
#        0.17478056, 0.8957839 , 0.99997103, 0.9999863 , 0.6363471 ,
#        0.19941382, 0.00360223, 0.99998045, 0.7757343 , 0.11905619,
#        0.2755659 , 0.85795456, 0.72782195, 0.21554251, 0.99814296,
#        0.99989843, 0.3164848 , 0.07685684, 0.08134694, 0.86106616,
#        0.9998523 , 0.9988023]
# label=[1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0,
#        1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1,
#        0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1,
#        1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]
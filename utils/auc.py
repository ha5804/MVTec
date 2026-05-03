from sklearn.metrics import roc_auc_score

def get_image_auc(test_loader, model):
    y_true = []
    y_score = []

    for img, label in test_loader:
        score, _ = model.predict(img)

        # label은 이미 int
        y_true.append(0 if label != 0 else 1)

        # score는 tensor → float
        y_score.append(score.detach().cpu().item())

    return roc_auc_score(y_true, y_score)

# def get_pixel_auc(test_loader, model):
#     y_true = []
#     y_score = []
#     for img, mask in test_loader:
#         _, heatmap = model.predict(img)

#         y_true.extend(mask.flatten())
#         y_score.extend(heatmap.flatten().cpu().item())
#     return roc_auc_score(y_true, y_score)
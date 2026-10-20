import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# =============================================================================
# 1. KLASIK YAPAY SINIR AGI (KARA KUTU / STANDARD MLP)
# =============================================================================
class StandardMLP(nn.Module):
    def __init__(self, in_features, hidden_nodes=10):
        super(StandardMLP, self).__init__()
        self.layer1 = nn.Linear(in_features, hidden_nodes)
        self.layer2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.layer3 = nn.Linear(hidden_nodes, 1)

    def forward(self, x):
        out = torch.relu(self.layer1(x))
        out = torch.relu(self.layer2(out))
        out = torch.sigmoid(self.layer3(out))
        return out

# =============================================================================
# 2. ONERILEN FS-ANN MODELI (T-NORM / T-CONORM & GAUSS UYELIK FONKSIYONU)
# =============================================================================
class FSANN_Fuzzy(nn.Module):
    def __init__(self, input_dim):
        super(FSANN_Fuzzy, self).__init__()
        # Oznitelik agirliklari (W)
        self.weights = nn.Parameter(torch.rand(input_dim))
        
        # Gauss Aktivasyon Parametreleri (Merkez ve Genislik)
        self.c = nn.Parameter(torch.tensor([1.0]))      
        self.sigma = nn.Parameter(torch.tensor([0.5]))  

    def forward(self, x):
        # Degerlerin [0, 1] araliginda kalmasini saglama
        w = torch.clamp(torch.sigmoid(self.weights), 1e-6, 1.0 - 1e-6)
        x_c = torch.clamp(x, 1e-6, 1.0 - 1e-6)
        
        # T-NORM (Cebirsel Carpim Kesisimi): x * w
        t_norm = x_c * w
        
        # T-CONORM (Cebirsel Toplam Birlesimi): 1 - prod(1 - T_norm)
        n = 1.0 - torch.prod(1.0 - t_norm, dim=1)
        
        # GAUSS AKTIVASYONU: exp( - (n - c)^2 / (2 * sigma^2) )
        sigma_sq = torch.clamp(self.sigma ** 2, min=1e-4)
        mu = torch.exp(-((n - self.c)**2) / (2 * sigma_sq))
        
        return mu.unsqueeze(1)

# =============================================================================
# 3. ORTAK EGITIM VE TEST FONKSIYONU
# =============================================================================
def egit_ve_test_et(model, model_adi, X_train, y_train, X_test, y_test, epoch_sayisi=500, lr=0.01):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoch_sayisi):
        model.train()
        optimizer.zero_grad()
        mu = model(X_train)
        loss = criterion(mu, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = (model(X_test) >= 0.5).int().numpy()
        y_true = y_test.numpy()

    print(f"\n[{model_adi} - Test Sonuclari]")
    print(f"Dogruluk (Accuracy) : % {accuracy_score(y_true, preds) * 100:.2f}")
    print(f"F1 Skoru            : {f1_score(y_true, preds, zero_division=0):.4f}")

    return model

# =============================================================================
# 4. UYGULAMA (GOGUS KANSERI VERI SETI)
# =============================================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    data = load_breast_cancer()
    X_raw = data.data
    
    # Kanser bagimsiz degiskeninin 1 olarak ayarlanmasi (0=Malignant, 1=Benign -> Ters Cevirme)
    y = 1 - data.target 
    ozellik_isimleri = data.feature_names

    # Anlamli ilk 5 ozelligin secimi
    selector = SelectKBest(score_func=f_classif, k=5)
    X_secilen = selector.fit_transform(X_raw, y)

    secilen_indeksler = selector.get_support(indices=True)
    secilen_isimler = [ozellik_isimleri[i] for i in secilen_indeksler]

    # KLASIK MLP ICIN STANDART OLCEKLEME
    scaler = MinMaxScaler()
    X_scaled_mlp = scaler.fit_transform(X_secilen)
    
    # =========================================================================
    # FS-ANN ICIN VERI ON ISLEME VE YON HIZALAMASI (SEMANTIC ALIGNMENT)
    # =========================================================================
    X_scaled_fsann = X_scaled_mlp.copy()
    for i in range(X_scaled_fsann.shape[1]):
        corr = np.corrcoef(X_scaled_fsann[:, i], y)[0, 1]
        if corr < 0:
            # Ters orantili ozelliklerin (1 - x) formulu ile duzeltilmesi
            X_scaled_fsann[:, i] = 1.0 - X_scaled_fsann[:, i]
    # =========================================================================

    X_train_m, X_test_m, y_train, y_test = train_test_split(X_scaled_mlp, y, test_size=0.2, random_state=42)
    X_train_f, X_test_f, _, _ = train_test_split(X_scaled_fsann, y, test_size=0.2, random_state=42)

    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    X_train_mlp_t = torch.tensor(X_train_m, dtype=torch.float32)
    X_test_mlp_t = torch.tensor(X_test_m, dtype=torch.float32)
    
    X_train_fsann_t = torch.tensor(X_train_f, dtype=torch.float32)
    X_test_fsann_t = torch.tensor(X_test_f, dtype=torch.float32)

    input_dim = 5

    # 1. Model Egitimi: MLP
    model_mlp = StandardMLP(input_dim)
    egit_ve_test_et(model_mlp, "Klasik MLP (Kara Kutu)", X_train_mlp_t, y_train_t, X_test_mlp_t, y_test_t, epoch_sayisi=800, lr=0.01)

    # 2. Model Egitimi: FS-ANN
    model_fsann = FSANN_Fuzzy(input_dim)
    egitilmis_fsann = egit_ve_test_et(model_fsann, "FS-ANN (Beyaz Kutu)", X_train_fsann_t, y_train_t, X_test_fsann_t, y_test_t, epoch_sayisi=800, lr=0.05)

    # ACIKLANABILIRLIK (XAI) ANALIZI
    print("\n--- ACIKLANABILIRLIK (XAI) VE KURAL CIKARIMI ---")
    
    print("\nKlasik MLP Agirlik Matrisi Ornegi (Yorumlanamaz):")
    print(np.round(model_mlp.layer1.weight.detach().numpy()[0][:5], 2))

    print("\nFS-ANN Karar Kurallari ve Parametre Agirliklari (W):")
    ogrenilen_agirliklar = torch.sigmoid(egitilmis_fsann.weights).detach().numpy()
    toplam_etki = np.sum(ogrenilen_agirliklar)
    
    for i, isim in enumerate(secilen_isimler):
        etki_yuzdesi = (ogrenilen_agirliklar[i] / toplam_etki) * 100
        print(f"[{isim}] Parametresinin T-Norm Etkisi : W = {ogrenilen_agirliklar[i]:.2f} (%{etki_yuzdesi:.1f})")
        
    ogrenilen_merkez = egitilmis_fsann.c.detach().numpy()[0]
    ogrenilen_genislik = egitilmis_fsann.sigma.detach().numpy()[0]
    
    print("\nOgrenilen Gauss Uyelik Fonksiyonu Degerleri:")
    print(f"Risk Zirvesi (Merkez c): {ogrenilen_merkez:.3f}")
    print(f"Tolerans Haddi (Sigma): {abs(ogrenilen_genislik):.3f}")

if __name__ == "__main__":
    main()

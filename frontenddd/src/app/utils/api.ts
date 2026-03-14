export interface UserProfile {
  allergies: string[];
  diseases: string[];
}

export interface IngredientRisk {
  ingredient: string;
  hazard_level: "safe" | "caution" | "hazard";
  explanation: string;
  related_allergies: string[];
  related_diseases: string[];
}

export interface ProductAnalysisResponse {
  product_name: string;
  barcode: string;
  category: string;
  ingredients: string[];
  ingredient_risks: IngredientRisk[];
  summary: string;
  warnings: string[];
  raw_openfoodfacts?: any;
}

export interface NewsArticle {
  title: string;
  description: string;
  url: string;
  source: string;
  publishedAt: string;
}

export interface ProductNewsResponse {
  articles: NewsArticle[];
}

// Helper to get saved profile from localStorage
export const getSavedProfile = (): UserProfile => {
  const profileStr = localStorage.getItem('wisebite_profile');
  if (profileStr) {
    try {
      return JSON.parse(profileStr);
    } catch (e) {
      console.error("Failed to parse profile", e);
    }
  }
  return { allergies: [], diseases: [] };
};

// Alias for SettingsPage usage
export const getUserProfile = getSavedProfile;

// Save profile to localStorage
export const saveUserProfile = (profile: UserProfile): void => {
  localStorage.setItem('wisebite_profile', JSON.stringify(profile));
};

// --- Recent Scans History ---
export interface RecentScan {
  id: string;
  product_name: string;
  category: string;
  summary: string;
  healthScore: number;
  hazardCount: number;
  cautionCount: number;
  safeCount: number;
  warningCount: number;
  scanDate: string;
  barcode: string;
  analysisResult: ProductAnalysisResponse;
}

const RECENT_SCANS_KEY = 'wisebite_recent_scans';
const MAX_RECENT_SCANS = 20;

export const saveRecentScan = (result: ProductAnalysisResponse): void => {
  const scans = getRecentScans();

  let score = 100;
  result.ingredient_risks.forEach(r => {
    if (r.hazard_level === 'hazard') score -= 20;
    else if (r.hazard_level === 'caution') score -= 10;
  });
  score = Math.max(0, score);

  const scan: RecentScan = {
    id: `${Date.now()}-${Math.random().toString(36).substring(2, 8)}`,
    product_name: result.product_name,
    category: result.category,
    summary: result.summary,
    healthScore: score,
    hazardCount: result.ingredient_risks.filter(r => r.hazard_level === 'hazard').length,
    cautionCount: result.ingredient_risks.filter(r => r.hazard_level === 'caution').length,
    safeCount: result.ingredient_risks.filter(r => r.hazard_level === 'safe').length,
    warningCount: result.warnings.length,
    scanDate: new Date().toISOString(),
    barcode: result.barcode || 'IMAGE_SCAN',
    analysisResult: result,
  };

  scans.unshift(scan);
  if (scans.length > MAX_RECENT_SCANS) scans.pop();
  localStorage.setItem(RECENT_SCANS_KEY, JSON.stringify(scans));
};

export const getRecentScans = (): RecentScan[] => {
  const str = localStorage.getItem(RECENT_SCANS_KEY);
  if (str) {
    try { return JSON.parse(str); } catch { return []; }
  }
  return [];
};

export const clearRecentScans = (): void => {
  localStorage.removeItem(RECENT_SCANS_KEY);
};

// 1. Analyze Label via Image Upload
export const analyzeLabelImage = async (imageFile: File, userProfile?: UserProfile): Promise<ProductAnalysisResponse> => {
  const profile = userProfile || getSavedProfile();
  
  const formData = new FormData();
  formData.append("label_image", imageFile);
  formData.append("allergies", profile.allergies.join(","));
  formData.append("diseases", profile.diseases.join(","));

  const response = await fetch('/api/analyze-label', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to analyze image (Status: ${response.status})`);
  }

  return response.json();
};

// 2. Extract Barcode from Image
export const extractBarcodeFromImage = async (imageFile: File): Promise<string> => {
  const formData = new FormData();
  formData.append("image", imageFile);

  const response = await fetch('/api/extract-barcode', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to extract barcode (Status: ${response.status})`);
  }

  const data = await response.json();
  return data.barcode;
};

// 3. Analyze Product by Barcode
export const analyzeProduct = async (barcode: string, userProfile?: UserProfile): Promise<ProductAnalysisResponse> => {
  const profile = userProfile || getSavedProfile();

  const response = await fetch('/api/analyze', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      barcode,
      user_profile: profile
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to analyze product (Status: ${response.status})`);
  }

  return response.json();
};

// 4. Get Product News
export const getProductNews = async (query: string): Promise<ProductNewsResponse> => {
  const response = await fetch(`/api/product-news?query=${encodeURIComponent(query)}`);
  
  if (!response.ok) {
    throw new Error(`Failed to fetch product news (Status: ${response.status})`);
  }

  return response.json();
};

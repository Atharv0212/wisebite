import { useEffect, useState } from "react";
import { useParams, Link } from "react-router";
import { Navigation } from "../components/Navigation";
import { BlossomDecoration } from "../components/BlossomDecoration";
import { HealthScoreGauge } from "../components/HealthScoreGauge";
import { ArrowLeft, AlertTriangle, Info, ShieldAlert, Leaf, Package, Sparkles } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "../components/ui/alert";
import { mockProducts } from "../data/mockData";
import { analyzeProduct } from "../api/client";
import type { ProductAnalysisResponse, IngredientRisk } from "../api/types";

export function ProductAnalysis() {
  const { id } = useParams();

  const [analysis, setAnalysis] = useState<ProductAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;

    const run = async () => {
      try {
        setLoading(true);
        setError(null);

        // Simple mapping from demo health profile into backend UserProfile
        const allergies = ["Tree Nuts", "Peanuts"];
        const diseases = ["Type 2 Diabetes"];

        const result = await analyzeProduct(id, { allergies, diseases });
        setAnalysis(result);
      } catch (e: any) {
        setError(e.message ?? "Failed to analyze product.");
      } finally {
        setLoading(false);
      }
    };

    void run();
  }, [id]);

  const fallbackProduct = analysis ? {
    id: analysis.barcode,
    name: analysis.product_name || "Unknown Product",
    brand: "Scanned Product",
    image: "https://images.unsplash.com/photo-1606859191214-25806e8e2423?auto=format&fit=crop&w=800&q=80",
    healthScore: analysis.ingredient_risks.some(r => r.hazard_level === "hazard") ? 30 : analysis.ingredient_risks.some(r => r.hazard_level === "caution") ? 65 : 90,
    scanDate: new Date().toISOString(),
    alerts: analysis.warnings || [],
    packagingScore: 70,
    recallStatus: null,
    ingredients: analysis.ingredients.map(ing => {
      const risk = analysis.ingredient_risks.find(r => r.ingredient.toLowerCase() === ing.toLowerCase());
      if (risk) {
        return {
          name: ing,
          simplified: risk.explanation,
          status: risk.hazard_level === "hazard" ? "danger" : risk.hazard_level === "caution" ? "warning" : "safe",
          reason: risk.related_allergies.length > 0 || risk.related_diseases.length > 0 ? "Conflicts with health profile" : undefined
        };
      }
      return {
        name: ing,
        simplified: "Standard ingredient",
        status: "safe"
      };
    })
  } : null;

  const product = mockProducts.find((p) => p.id === id) || fallbackProduct;

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-pink-50 via-purple-50 to-white">
        <BlossomDecoration />
        <Navigation />
        <div className="flex flex-col items-center justify-center min-h-[60vh]">
          <div className="animate-spin text-5xl mb-4">🌸</div>
          <p className="text-xl text-pink-600 font-medium tracking-wide">Analyzing ingredients...</p>
        </div>
      </div>
    );
  }

  if (!product) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-pink-50 via-purple-50 to-white">
        <BlossomDecoration />
        <Navigation />
        <div className="max-w-7xl mx-auto px-4 py-8">
          <p>Product not found</p>
        </div>
      </div>
    );
  }

  const backendHazards: IngredientRisk[] =
    analysis?.ingredient_risks.filter(
      (risk) => risk.hazard_level === "caution" || risk.hazard_level === "hazard"
    ) ?? [];

  const personalizedHazards =
    backendHazards.length > 0
      ? backendHazards.map((risk) => ({
          name: risk.ingredient,
          simplified: risk.explanation,
          status: risk.hazard_level === "hazard" ? "danger" : "warning",
          reason:
            risk.related_allergies.length > 0 || risk.related_diseases.length > 0
              ? `Linked to: ${[
                  ...risk.related_allergies,
                  ...risk.related_diseases,
                ].join(", ")}`
              : undefined,
        }))
      : product.ingredients.filter(
          (ing) => ing.status === "warning" || ing.status === "danger"
        );

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-50 via-purple-50 to-white relative overflow-hidden">
      <BlossomDecoration />
      <Navigation />
      
      {/* Decorative elements */}
      <div className="absolute top-20 right-0 w-80 h-80 opacity-10 pointer-events-none">
        <img 
          src="https://images.unsplash.com/photo-1651472989304-a2b2390617ca?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjaGVycnklMjBibG9zc29tJTIwc2FrdXJhJTIwcGluayUyMGZsb3dlcnN8ZW58MXx8fHwxNzczNDA1MDY0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
          alt=""
          className="w-full h-full object-cover"
        />
      </div>
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
        {/* Back Button */}
        <Link to="/">
          <Button variant="ghost" className="mb-6 hover:bg-pink-50 text-pink-600 rounded-full">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Dashboard
          </Button>
        </Link>

        {/* Product Header */}
        <div className="bg-gradient-to-r from-pink-50 via-white to-purple-50 rounded-3xl shadow-xl border-4 border-pink-200 p-8 mb-6 overflow-hidden relative">
          {/* Decorative wave */}
          <div className="absolute inset-0 opacity-5 pointer-events-none">
            <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
              <pattern id="header-wave" x="0" y="0" width="200" height="40" patternUnits="userSpaceOnUse">
                <path d="M0 20 Q 50 10, 100 20 T 200 20" fill="none" stroke="#ec4899" strokeWidth="2"/>
              </pattern>
              <rect width="100%" height="100%" fill="url(#header-wave)" />
            </svg>
          </div>
          
          <div className="flex flex-col md:flex-row gap-6 relative">
            <div className="w-full md:w-64 h-64 bg-gradient-to-br from-pink-100 to-purple-100 rounded-2xl overflow-hidden flex-shrink-0 border-3 border-pink-200 shadow-lg">
              <img
                src={product.image}
                alt={product.name}
                className="w-full h-full object-cover"
              />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-3xl">🌸</span>
                <p className="text-sm text-pink-500 font-semibold">{product.brand}</p>
              </div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-pink-600 to-purple-600 bg-clip-text text-transparent mb-4">
                {product.name}
              </h1>
              <div className="flex flex-wrap gap-2">
                <Badge className="bg-gradient-to-r from-pink-100 to-purple-100 text-pink-700 border-2 border-pink-300">
                  Scanned {new Date(product.scanDate).toLocaleDateString()}
                </Badge>
                {product.recallStatus && (
                  <Badge className="bg-gradient-to-r from-red-400 to-pink-400 text-white border-none">
                    Recall Active
                  </Badge>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Overall Health Score */}
        <Card className="mb-6 border-4 border-pink-300 shadow-xl overflow-hidden">
          <div className="absolute inset-0 opacity-5 pointer-events-none">
            <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
              <pattern id="score-pattern" x="0" y="0" width="50" height="50" patternUnits="userSpaceOnUse">
                <circle cx="25" cy="25" r="2" fill="#ec4899"/>
                <circle cx="10" cy="10" r="1.5" fill="#a855f7"/>
              </pattern>
              <rect width="100%" height="100%" fill="url(#score-pattern)" />
            </svg>
          </div>
          <CardHeader className="bg-gradient-to-r from-pink-50 to-purple-50 border-b-2 border-pink-200">
            <CardTitle className="text-3xl flex items-center gap-2">
              <Sparkles className="w-7 h-7 text-pink-500 animate-sakura-bloom" />
              <span className="bg-gradient-to-r from-pink-600 to-purple-600 bg-clip-text text-transparent">
                Overall Health Assessment
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-8 relative">
            <div className="flex flex-col md:flex-row items-center gap-8">
              <div className="flex-shrink-0">
                <HealthScoreGauge score={product.healthScore} size="large" />
              </div>
              <div className="flex-1 space-y-4">
                <div className="bg-gradient-to-br from-pink-50 to-purple-50 p-6 rounded-2xl border-2 border-pink-200">
                  <h3 className="font-bold text-gray-900 mb-3 text-xl">Risk Assessment</h3>
                  <p className="text-gray-700 leading-relaxed">
                    This product has been analyzed against your personal health profile. 
                    {product.healthScore >= 80 && " This is a safe choice for your dietary needs. 🌸"}
                    {product.healthScore >= 60 && product.healthScore < 80 && " This product contains some ingredients to be aware of. 🌿"}
                    {product.healthScore < 60 && " This product contains ingredients that may pose risks based on your health profile. ⚠️"}
                  </p>
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-gradient-to-br from-pink-100 to-white p-4 rounded-2xl border-2 border-pink-200 text-center">
                    <p className="text-xs text-pink-600 font-semibold mb-1">Ingredients</p>
                    <p className="text-2xl font-bold text-gray-900">{product.ingredients.length}</p>
                  </div>
                  <div className="bg-gradient-to-br from-purple-100 to-white p-4 rounded-2xl border-2 border-purple-200 text-center">
                    <p className="text-xs text-purple-600 font-semibold mb-1">Packaging</p>
                    <p className="text-2xl font-bold text-gray-900">{product.packagingScore}/100</p>
                  </div>
                  <div className="bg-gradient-to-br from-pink-100 to-white p-4 rounded-2xl border-2 border-pink-200 text-center">
                    <p className="text-xs text-pink-600 font-semibold mb-1">Alerts</p>
                    <p className="text-2xl font-bold text-gray-900">{product.alerts.length}</p>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="grid lg:grid-cols-2 gap-6 mb-6">
          {/* Personalized Hazards */}
          <Card className="border-4 border-red-300 shadow-xl overflow-hidden">
            <CardHeader className="bg-gradient-to-br from-red-50 to-pink-50 border-b-2 border-red-200">
              <CardTitle className="flex items-center gap-2 text-xl">
                <ShieldAlert className="w-6 h-6 text-red-500" />
                <span className="bg-gradient-to-r from-red-600 to-pink-600 bg-clip-text text-transparent">
                  Personalized Hazards
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-6">
              {personalizedHazards.length > 0 ? (
                <div className="space-y-3">
                  {personalizedHazards.map((ingredient, idx) => (
                    <Alert key={idx} className="border-2 border-red-300 bg-gradient-to-r from-red-50 to-pink-50">
                      <AlertTriangle className="h-5 w-5 text-red-500" />
                      <AlertTitle className="text-red-800 font-bold">{ingredient.name}</AlertTitle>
                      <AlertDescription className="text-red-700">
                        {ingredient.reason || "This ingredient conflicts with your health profile"}
                      </AlertDescription>
                    </Alert>
                  ))}
                </div>
              ) : (
                <div className="flex items-center gap-3 text-green-600 bg-gradient-to-r from-green-50 to-emerald-50 p-6 rounded-2xl border-2 border-green-300">
                  <Info className="w-6 h-6" />
                  <p className="font-semibold">No personalized hazards detected 🌸</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Packaging Toxicity */}
          <Card className="border-4 border-purple-300 shadow-xl overflow-hidden">
            <CardHeader className="bg-gradient-to-br from-purple-50 to-pink-50 border-b-2 border-purple-200">
              <CardTitle className="flex items-center gap-2 text-xl">
                <Package className="w-6 h-6 text-purple-500" />
                <span className="bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                  Packaging Analysis
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between bg-gradient-to-r from-purple-50 to-pink-50 p-4 rounded-2xl border-2 border-purple-200">
                  <span className="font-semibold text-gray-700">Safety Score</span>
                  <span className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                    {product.packagingScore}/100
                  </span>
                </div>
                <div className="space-y-3 pt-2">
                  <div className="flex items-start gap-3 bg-white p-3 rounded-xl border border-green-200">
                    <div className="w-3 h-3 bg-green-400 rounded-full mt-1.5"></div>
                    <div>
                      <p className="font-semibold text-gray-900">BPA-Free</p>
                      <p className="text-sm text-gray-600">No bisphenol detected</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3 bg-white p-3 rounded-xl border border-green-200">
                    <div className="w-3 h-3 bg-green-400 rounded-full mt-1.5"></div>
                    <div>
                      <p className="font-semibold text-gray-900">Recyclable Material</p>
                      <p className="text-sm text-gray-600">Eco-friendly packaging</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3 bg-white p-3 rounded-xl border border-amber-200">
                    <div className="w-3 h-3 bg-amber-400 rounded-full mt-1.5"></div>
                    <div>
                      <p className="font-semibold text-gray-900">Handling Notice</p>
                      <p className="text-sm text-gray-600">Wash hands after handling</p>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Public Safety Banner */}
        {product.recallStatus && (
          <Alert className="mb-6 border-4 border-red-400 bg-gradient-to-r from-red-50 to-pink-50">
            <AlertTriangle className="h-6 w-6 text-red-500" />
            <AlertTitle className="text-xl text-red-800 font-bold">FDA Recall Notice</AlertTitle>
            <AlertDescription className="text-base text-red-700">
              This product is currently under an active recall. Do not consume and return to place of purchase.
            </AlertDescription>
          </Alert>
        )}

        {/* Translated Ingredients */}
        <Card className="border-4 border-pink-300 shadow-xl overflow-hidden">
          <CardHeader className="bg-gradient-to-r from-pink-50 via-purple-50 to-pink-50 border-b-2 border-pink-200">
            <CardTitle className="text-2xl flex items-center gap-2">
              <Leaf className="w-7 h-7 text-green-500" />
              <span className="bg-gradient-to-r from-pink-600 to-purple-600 bg-clip-text text-transparent">
                Translated Ingredients
              </span>
            </CardTitle>
            <p className="text-sm text-gray-600 mt-2">
              Complex ingredients explained in simple terms 🌸
            </p>
          </CardHeader>
          <CardContent className="p-6">
            <div className="space-y-4">
              {product.ingredients.map((ingredient, idx) => (
                <div
                  key={idx}
                  className={`p-5 rounded-2xl border-l-4 shadow-md ${
                    ingredient.status === "safe"
                      ? "bg-gradient-to-r from-green-50 to-emerald-50 border-l-green-500"
                      : ingredient.status === "warning"
                      ? "bg-gradient-to-r from-amber-50 to-yellow-50 border-l-amber-500"
                      : ingredient.status === "danger"
                      ? "bg-gradient-to-r from-red-50 to-pink-50 border-l-red-500"
                      : "bg-gradient-to-r from-gray-50 to-slate-50 border-l-gray-300"
                  }`}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <h4 className="font-bold text-gray-900 mb-2 text-lg">
                        {ingredient.name}
                      </h4>
                      <p className="text-gray-700 mb-2 leading-relaxed">
                        {ingredient.simplified}
                      </p>
                      {ingredient.reason && (
                        <p className="text-sm font-semibold text-red-700 flex items-center gap-1">
                          ⚠️ {ingredient.reason}
                        </p>
                      )}
                    </div>
                    <Badge
                      className={`${
                        ingredient.status === "safe"
                          ? "bg-gradient-to-r from-green-400 to-emerald-400 text-white border-none"
                          : ingredient.status === "warning"
                          ? "bg-gradient-to-r from-amber-400 to-yellow-400 text-white border-none"
                          : ingredient.status === "danger"
                          ? "bg-gradient-to-r from-red-400 to-pink-400 text-white border-none"
                          : "bg-gradient-to-r from-gray-400 to-slate-400 text-white border-none"
                      }`}
                    >
                      {ingredient.status === "safe" && "Safe"}
                      {ingredient.status === "warning" && "Caution"}
                      {ingredient.status === "danger" && "Risk"}
                      {ingredient.status === "caution" && "Monitor"}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
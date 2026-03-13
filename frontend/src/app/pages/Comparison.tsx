import { useState } from "react";
import { Navigation } from "../components/Navigation";
import { BlossomDecoration } from "../components/BlossomDecoration";
import { HealthScoreGauge } from "../components/HealthScoreGauge";
import { TrendingUp, AlertTriangle, Package, CheckCircle, Sparkles, X, Plus } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { mockProducts } from "../data/mockData";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "../components/ui/dialog";
import { getRecentScans } from "../utils/storage";

export function Comparison() {
  // Always get the latest scans on mount
  const recentScans = getRecentScans();
  
  // Default to the first two recent scans, or mock products if less than 2
  const defaultA = recentScans.length > 0 ? recentScans[0] : mockProducts[1];
  const defaultB = recentScans.length > 1 ? recentScans[1] : mockProducts[2];

  const [selectedProductA, setSelectedProductA] = useState(defaultA);
  const [selectedProductB, setSelectedProductB] = useState(defaultB);
  const [isDialogAOpen, setIsDialogAOpen] = useState(false);
  const [isDialogBOpen, setIsDialogBOpen] = useState(false);

  const winner = selectedProductA.healthScore > selectedProductB.healthScore ? selectedProductA : selectedProductB;

  const ProductSelector = ({ 
    onSelect, 
    currentProductId, 
    position 
  }: { 
    onSelect: (product: typeof mockProducts[0]) => void;
    currentProductId: string;
    position: "A" | "B";
  }) => (
    <div className="grid grid-cols-2 gap-4 max-h-96 overflow-y-auto">
      {recentScans
        .filter((p: any) => p.id !== currentProductId)
        .map((product: any) => (
          <Card
            key={product.id}
            className="cursor-pointer hover:shadow-lg transition-all border-2 border-pink-200 hover:border-pink-400"
            onClick={() => {
              onSelect(product);
              if (position === "A") setIsDialogAOpen(false);
              else setIsDialogBOpen(false);
            }}
          >
            <CardContent className="p-3">
              <div className="aspect-square bg-gradient-to-br from-pink-50 to-purple-50 rounded-lg mb-2 overflow-hidden">
                <img
                  src={product.image}
                  alt={product.name}
                  className="w-full h-full object-cover"
                />
              </div>
              <h4 className="font-semibold text-sm text-gray-900 line-clamp-2 mb-1">
                {product.name}
              </h4>
              <p className="text-xs text-gray-500">{product.brand}</p>
              <div className="mt-2 flex items-center justify-between">
                <span className="text-xs text-gray-500">Score:</span>
                <span
                  className="text-lg font-bold"
                  style={{
                    color:
                      product.healthScore >= 80
                        ? "#10b981"
                        : product.healthScore >= 60
                        ? "#f59e0b"
                        : "#ef4444",
                  }}
                >
                  {product.healthScore}
                </span>
              </div>
            </CardContent>
          </Card>
        ))}
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-50 via-purple-50 to-white relative overflow-hidden">
      <BlossomDecoration />
      <Navigation />
      
      {/* Decorative elements */}
      <div className="absolute top-20 left-0 w-64 h-64 opacity-10 pointer-events-none">
        <img 
          src="https://images.unsplash.com/photo-1651472989304-a2b2390617ca?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjaGVycnklMjBibG9zc29tJTIwc2FrdXJhJTIwcGluayUyMGZsb3dlcnN8ZW58MXx8fHwxNzczNDA1MDY0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
          alt=""
          className="w-full h-full object-cover"
        />
      </div>
      <div className="absolute bottom-20 right-0 w-64 h-64 opacity-10 pointer-events-none">
        <img 
          src="https://images.unsplash.com/photo-1651472989304-a2b2390617ca?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjaGVycnklMjBibG9zc29tJTIwc2FrdXJhJTIwcGluayUyMGZsb3dlcnN8ZW58MXx8fHwxNzczNDA1MDY0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
          alt=""
          className="w-full h-full object-cover transform -scale-x-100"
        />
      </div>
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
        <div className="mb-8 text-center">
          <div className="inline-block mb-4">
            <div className="flex items-center gap-2 bg-gradient-to-r from-pink-100 to-purple-100 px-6 py-2 rounded-full border-2 border-pink-300 shadow-lg">
              <Sparkles className="w-5 h-5 text-pink-500 animate-sakura-bloom" />
              <span className="text-sm font-semibold bg-gradient-to-r from-pink-600 to-purple-600 bg-clip-text text-transparent">
                Make the Wise Choice
              </span>
            </div>
          </div>
          <h1 className="text-5xl font-bold bg-gradient-to-r from-pink-600 via-purple-600 to-pink-600 bg-clip-text text-transparent mb-2">
            Product Comparison
          </h1>
          <p className="text-xl text-gray-600">
            Side-by-side analysis to guide your decision 🌸
          </p>
        </div>

        {/* Comparison Grid */}
        <div className="grid lg:grid-cols-2 gap-6 mb-6">
          {/* Product A */}
          <Card className={`border-4 ${winner.id === selectedProductA.id ? 'border-green-400 shadow-2xl' : 'border-pink-300'} overflow-hidden`}>
            <CardHeader className="bg-gradient-to-br from-pink-100 via-purple-100 to-pink-100 relative">
              {/* Decorative pattern */}
              <div className="absolute inset-0 opacity-10 pointer-events-none">
                <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
                  <pattern id="pattern-a" x="0" y="0" width="40" height="40" patternUnits="userSpaceOnUse">
                    <circle cx="20" cy="20" r="2" fill="#ec4899"/>
                  </pattern>
                  <rect width="100%" height="100%" fill="url(#pattern-a)" />
                </svg>
              </div>
              
              <div className="flex justify-between items-start mb-4 relative">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <Badge className="bg-gradient-to-r from-pink-500 to-purple-500 text-white border-none">
                      Product A
                    </Badge>
                    {winner.id === selectedProductA.id && (
                      <Badge className="bg-gradient-to-r from-green-400 to-emerald-400 text-white border-none shadow-lg">
                        <TrendingUp className="w-3 h-3 mr-1" />
                        Winner
                      </Badge>
                    )}
                  </div>
                  <p className="text-sm text-gray-600 mb-1">{selectedProductA.brand}</p>
                  <CardTitle className="text-2xl">{selectedProductA.name}</CardTitle>
                </div>
                
                <Dialog open={isDialogAOpen} onOpenChange={setIsDialogAOpen}>
                  <DialogTrigger asChild>
                    <Button 
                      variant="outline" 
                      size="sm"
                      className="border-pink-300 text-pink-600 hover:bg-pink-50 rounded-full"
                    >
                      Change
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-3xl bg-gradient-to-br from-pink-50 to-white border-4 border-pink-300">
                    <DialogHeader>
                      <DialogTitle className="text-2xl bg-gradient-to-r from-pink-600 to-purple-600 bg-clip-text text-transparent">
                        Select Product A
                      </DialogTitle>
                      <DialogDescription>
                        Choose a product to compare
                      </DialogDescription>
                    </DialogHeader>
                    <ProductSelector 
                      onSelect={setSelectedProductA} 
                      currentProductId={selectedProductB.id}
                      position="A"
                    />
                  </DialogContent>
                </Dialog>
              </div>
              
              <div className="aspect-video bg-gradient-to-br from-pink-50 to-purple-50 rounded-2xl overflow-hidden border-2 border-pink-200 shadow-inner">
                <img
                  src={selectedProductA.image}
                  alt={selectedProductA.name}
                  className="w-full h-full object-cover"
                />
              </div>
            </CardHeader>
            <CardContent className="pt-6 bg-gradient-to-br from-white to-pink-50">
              <div className="flex justify-center mb-6">
                <HealthScoreGauge score={selectedProductA.healthScore} size="small" />
              </div>

              <div className="space-y-4">
                <div className="bg-white p-4 rounded-2xl border-2 border-pink-100">
                  <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                    <AlertTriangle className="w-5 h-5 text-pink-500" />
                    Allergen Alerts
                  </h4>
                  {selectedProductA.alerts.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                      {selectedProductA.alerts.map((alert: string, idx: number) => (
                        <Badge key={idx} className="bg-gradient-to-r from-red-400 to-pink-400 text-white border-none">
                          {alert}
                        </Badge>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-green-600 flex items-center gap-1">
                      <CheckCircle className="w-4 h-4" />
                      No allergen alerts
                    </p>
                  )}
                </div>

                <div className="bg-white p-4 rounded-2xl border-2 border-pink-100">
                  <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                    <Package className="w-5 h-5 text-purple-500" />
                    Packaging Safety
                  </h4>
                  <div className="flex items-center gap-3">
                    <div className="flex-1 bg-gradient-to-r from-pink-100 to-purple-100 rounded-full h-3">
                      <div
                        className="bg-gradient-to-r from-purple-500 to-pink-500 h-3 rounded-full"
                        style={{ width: `${selectedProductA.packagingScore}%` }}
                      />
                    </div>
                    <span className="text-lg font-bold bg-gradient-to-r from-pink-600 to-purple-600 bg-clip-text text-transparent">
                      {selectedProductA.packagingScore}/100
                    </span>
                  </div>
                </div>

                <div className="bg-white p-4 rounded-2xl border-2 border-pink-100">
                  <h4 className="font-semibold text-gray-900 mb-3">Key Ingredients</h4>
                  <div className="space-y-2">
                    {selectedProductA.ingredients.slice(0, 3).map((ing: any, idx: number) => (
                      <div key={idx} className="flex items-center gap-2">
                        <div
                          className={`w-3 h-3 rounded-full ${
                            ing.status === "safe"
                              ? "bg-green-400"
                              : ing.status === "warning"
                              ? "bg-amber-400"
                              : "bg-red-400"
                          }`}
                        />
                        <span className="text-sm text-gray-700">{ing.name}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Product B */}
          <Card className={`border-4 ${winner.id === selectedProductB.id ? 'border-green-400 shadow-2xl' : 'border-purple-300'} overflow-hidden`}>
            <CardHeader className="bg-gradient-to-br from-purple-100 via-pink-100 to-purple-100 relative">
              {/* Decorative pattern */}
              <div className="absolute inset-0 opacity-10 pointer-events-none">
                <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
                  <pattern id="pattern-b" x="0" y="0" width="40" height="40" patternUnits="userSpaceOnUse">
                    <circle cx="20" cy="20" r="2" fill="#a855f7"/>
                  </pattern>
                  <rect width="100%" height="100%" fill="url(#pattern-b)" />
                </svg>
              </div>
              
              <div className="flex justify-between items-start mb-4 relative">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <Badge className="bg-gradient-to-r from-purple-500 to-pink-500 text-white border-none">
                      Product B
                    </Badge>
                    {winner.id === selectedProductB.id && (
                      <Badge className="bg-gradient-to-r from-green-400 to-emerald-400 text-white border-none shadow-lg">
                        <TrendingUp className="w-3 h-3 mr-1" />
                        Winner
                      </Badge>
                    )}
                  </div>
                  <p className="text-sm text-gray-600 mb-1">{selectedProductB.brand}</p>
                  <CardTitle className="text-2xl">{selectedProductB.name}</CardTitle>
                </div>
                
                <Dialog open={isDialogBOpen} onOpenChange={setIsDialogBOpen}>
                  <DialogTrigger asChild>
                    <Button 
                      variant="outline" 
                      size="sm"
                      className="border-purple-300 text-purple-600 hover:bg-purple-50 rounded-full"
                    >
                      Change
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-3xl bg-gradient-to-br from-purple-50 to-white border-4 border-purple-300">
                    <DialogHeader>
                      <DialogTitle className="text-2xl bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                        Select Product B
                      </DialogTitle>
                      <DialogDescription>
                        Choose a product to compare
                      </DialogDescription>
                    </DialogHeader>
                    <ProductSelector 
                      onSelect={setSelectedProductB} 
                      currentProductId={selectedProductA.id}
                      position="B"
                    />
                  </DialogContent>
                </Dialog>
              </div>
              
              <div className="aspect-video bg-gradient-to-br from-purple-50 to-pink-50 rounded-2xl overflow-hidden border-2 border-purple-200 shadow-inner">
                <img
                  src={selectedProductB.image}
                  alt={selectedProductB.name}
                  className="w-full h-full object-cover"
                />
              </div>
            </CardHeader>
            <CardContent className="pt-6 bg-gradient-to-br from-white to-purple-50">
              <div className="flex justify-center mb-6">
                <HealthScoreGauge score={selectedProductB.healthScore} size="small" />
              </div>

              <div className="space-y-4">
                <div className="bg-white p-4 rounded-2xl border-2 border-purple-100">
                  <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                    <AlertTriangle className="w-5 h-5 text-purple-500" />
                    Allergen Alerts
                  </h4>
                  {selectedProductB.alerts.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                      {selectedProductB.alerts.map((alert: string, idx: number) => (
                        <Badge key={idx} className="bg-gradient-to-r from-red-400 to-pink-400 text-white border-none">
                          {alert}
                        </Badge>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-green-600 flex items-center gap-1">
                      <CheckCircle className="w-4 h-4" />
                      No allergen alerts
                    </p>
                  )}
                </div>

                <div className="bg-white p-4 rounded-2xl border-2 border-purple-100">
                  <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                    <Package className="w-5 h-5 text-pink-500" />
                    Packaging Safety
                  </h4>
                  <div className="flex items-center gap-3">
                    <div className="flex-1 bg-gradient-to-r from-purple-100 to-pink-100 rounded-full h-3">
                      <div
                        className="bg-gradient-to-r from-pink-500 to-purple-500 h-3 rounded-full"
                        style={{ width: `${selectedProductB.packagingScore}%` }}
                      />
                    </div>
                    <span className="text-lg font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                      {selectedProductB.packagingScore}/100
                    </span>
                  </div>
                </div>

                <div className="bg-white p-4 rounded-2xl border-2 border-purple-100">
                  <h4 className="font-semibold text-gray-900 mb-3">Key Ingredients</h4>
                  <div className="space-y-2">
                    {selectedProductB.ingredients.slice(0, 3).map((ing: any, idx: number) => (
                      <div key={idx} className="flex items-center gap-2">
                        <div
                          className={`w-3 h-3 rounded-full ${
                            ing.status === "safe"
                              ? "bg-green-400"
                              : ing.status === "warning"
                              ? "bg-amber-400"
                              : "bg-red-400"
                          }`}
                        />
                        <span className="text-sm text-gray-700">{ing.name}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* AI Verdict */}
        <Card className="border-4 border-pink-300 overflow-hidden shadow-2xl">
          <div className="absolute inset-0 opacity-5 pointer-events-none">
            <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
              <pattern id="verdict-wave" x="0" y="0" width="200" height="60" patternUnits="userSpaceOnUse">
                <path d="M0 30 Q 50 20, 100 30 T 200 30" fill="none" stroke="#ec4899" strokeWidth="3"/>
              </pattern>
              <rect width="100%" height="100%" fill="url(#verdict-wave)" />
            </svg>
          </div>
          
          <CardHeader className="bg-gradient-to-r from-pink-100 via-purple-100 to-pink-100 relative border-b-2 border-pink-200">
            <CardTitle className="flex items-center gap-3 text-3xl">
              <Sparkles className="w-8 h-8 text-pink-500 animate-sakura-bloom" />
              <span className="bg-gradient-to-r from-pink-600 to-purple-600 bg-clip-text text-transparent">
                AI Verdict
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-8 relative">
            <div className="space-y-6">
              <div className="bg-gradient-to-br from-white to-pink-50 rounded-3xl p-8 border-3 border-pink-200 shadow-lg">
                <div className="flex items-center gap-3 mb-4">
                  <span className="text-4xl">🌸</span>
                  <h3 className="text-2xl font-bold bg-gradient-to-r from-pink-600 to-purple-600 bg-clip-text text-transparent">
                    Recommended: {winner.name}
                  </h3>
                </div>
                <p className="text-gray-700 text-lg leading-relaxed mb-6">
                  Based on your health profile and our comprehensive analysis, <strong>{winner.name}</strong> by {winner.brand} is the wiser choice for your wellbeing.
                </p>
                
                <div className="grid md:grid-cols-3 gap-6 pt-6 border-t-2 border-pink-200">
                  <div className="text-center bg-gradient-to-br from-pink-100 to-purple-100 p-4 rounded-2xl border-2 border-pink-200">
                    <p className="text-sm font-semibold text-gray-600 mb-2">Health Score Difference</p>
                    <p className="text-4xl font-bold bg-gradient-to-r from-pink-600 to-purple-600 bg-clip-text text-transparent">
                      +{Math.abs(selectedProductA.healthScore - selectedProductB.healthScore)}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">points</p>
                  </div>
                  <div className="text-center bg-gradient-to-br from-purple-100 to-pink-100 p-4 rounded-2xl border-2 border-purple-200">
                    <p className="text-sm font-semibold text-gray-600 mb-2">Fewer Alerts</p>
                    <p className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                      {Math.max(selectedProductA.alerts.length, selectedProductB.alerts.length) - Math.min(selectedProductA.alerts.length, selectedProductB.alerts.length)}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">less warnings</p>
                  </div>
                  <div className="text-center bg-gradient-to-br from-pink-100 to-purple-100 p-4 rounded-2xl border-2 border-pink-200">
                    <p className="text-sm font-semibold text-gray-600 mb-2">Packaging Quality</p>
                    <p className="text-4xl font-bold bg-gradient-to-r from-pink-600 to-purple-600 bg-clip-text text-transparent">
                      {winner.packagingScore > (winner === selectedProductA ? selectedProductB.packagingScore : selectedProductA.packagingScore) ? 'Better' : 'Good'}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">rating</p>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-2xl p-6 border-2 border-purple-200">
                <h4 className="font-bold text-gray-900 mb-4 text-lg flex items-center gap-2">
                  <span>💡</span>
                  Why this recommendation?
                </h4>
                <ul className="space-y-3 text-gray-700">
                  {winner.id === selectedProductA.id ? (
                    <>
                      <li className="flex items-start gap-3">
                        <CheckCircle className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                        <span>Significantly higher overall health score ({selectedProductA.healthScore} vs {selectedProductB.healthScore})</span>
                      </li>
                      <li className="flex items-start gap-3">
                        <CheckCircle className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                        <span>Contains natural, minimally processed ingredients</span>
                      </li>
                      <li className="flex items-start gap-3">
                        <CheckCircle className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                        <span>Better aligns with your dietary preferences and health conditions</span>
                      </li>
                    </>
                  ) : (
                    <>
                      <li className="flex items-start gap-3">
                        <CheckCircle className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                        <span>Higher overall health score ({selectedProductB.healthScore} vs {selectedProductA.healthScore})</span>
                      </li>
                      <li className="flex items-start gap-3">
                        <CheckCircle className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                        <span>Fewer allergen warnings based on your profile</span>
                      </li>
                      <li className="flex items-start gap-3">
                        <CheckCircle className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                        <span>Better packaging safety and ingredient transparency</span>
                      </li>
                    </>
                  )}
                </ul>
              </div>

              <div className="flex justify-center pt-4">
                <Button size="lg" className="bg-gradient-to-r from-pink-500 via-purple-500 to-pink-500 hover:from-pink-600 hover:via-purple-600 hover:to-pink-600 px-12 py-6 text-lg rounded-full shadow-xl">
                  View {winner.name} Details 🌸
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
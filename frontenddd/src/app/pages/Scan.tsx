import { Link, useNavigate } from "react-router";
import { WiseBiteLogo } from "../components/WiseBiteLogo";
import { Home as HomeIcon, GitCompare, User, Menu, Settings, ScanLine, History, Camera, Upload } from "lucide-react";
import { Card, CardContent } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { useState, useRef } from "react";
import { analyzeLabelImage, extractBarcodeFromImage, analyzeProduct, ProductAnalysisResponse, getUserProfile, saveRecentScan } from "../utils/api";

export function Scan() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStep, setAnalysisStep] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    
    setIsAnalyzing(true);
    const profile = getUserProfile();

    try {
      // STEP 1: Try to extract a barcode from the image
      setAnalysisStep("🔍 Step 1/3: Searching for barcode in image...");
      let barcodeResult: string | null = null;
      try {
        barcodeResult = await extractBarcodeFromImage(selectedFile);
      } catch {
        // No barcode found — this is expected, continue to fallback
        barcodeResult = null;
      }

      if (barcodeResult) {
        // STEP 2a: Barcode found! Look up product via OpenFoodFacts + Gemini
        setAnalysisStep(`✅ Barcode found: ${barcodeResult}. Looking up product database...`);
        try {
          const result: ProductAnalysisResponse = await analyzeProduct(barcodeResult, profile);
          saveRecentScan(result);
          navigate(`/product/${barcodeResult}`, { state: { analysisResult: result } });
          return;
        } catch {
          // Product not in OpenFoodFacts — fall through to image analysis
          setAnalysisStep("⚠️ Product not found in database. Analyzing image directly...");
        }
      }

      // STEP 2b/3: No barcode (or barcode product not found) — analyze the image directly
      setAnalysisStep("🧪 Step 2/3: Analyzing ingredient list from image...");
      const result: ProductAnalysisResponse = await analyzeLabelImage(selectedFile, profile);
      saveRecentScan(result);
      navigate(`/product/${result.barcode || 'IMAGE_SCAN'}`, { state: { analysisResult: result } });

    } catch (error) {
      console.error("Analysis failed:", error);
      alert(error instanceof Error ? error.message : "Analysis failed. Please try again.");
    } finally {
      setIsAnalyzing(false);
      setAnalysisStep("");
    }
  };

  return (
    <div className="min-h-screen relative overflow-hidden" style={{ backgroundColor: '#E0FFF5' }}>
      {/* Left Sidebar */}
      <div 
        className={`fixed left-0 top-0 h-full bg-white shadow-xl z-50 transition-all duration-300 ${
          sidebarOpen ? 'w-64' : 'w-20'
        }`}
      >
        <div className="p-4 border-b border-emerald-200 flex items-center justify-between">
          {sidebarOpen && (
            <span className="text-lg font-bold bg-gradient-to-r from-emerald-600 to-green-600 bg-clip-text text-transparent">
              Menu
            </span>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="hover:bg-emerald-50"
          >
            <Menu className="w-5 h-5 text-emerald-600" />
          </Button>
        </div>

        <nav className="p-4 space-y-2">
          <Link to="/home">
            <div className="flex items-center gap-3 p-3 rounded-lg hover:bg-emerald-50 text-gray-700 cursor-pointer transition-all">
              <HomeIcon className="w-5 h-5 text-emerald-600" />
              {sidebarOpen && <span className="font-medium">Home</span>}
            </div>
          </Link>

          <Link to="/scan">
            <div className="flex items-center gap-3 p-3 rounded-lg bg-gradient-to-r from-emerald-500 to-emerald-600 text-white shadow-md cursor-pointer">
              <ScanLine className="w-5 h-5" />
              {sidebarOpen && <span className="font-medium">Scan</span>}
            </div>
          </Link>

          <Link to="/recent-scans">
            <div className="flex items-center gap-3 p-3 rounded-lg hover:bg-emerald-50 text-gray-700 cursor-pointer transition-all">
              <History className="w-5 h-5 text-emerald-600" />
              {sidebarOpen && <span className="font-medium">Recent Scans</span>}
            </div>
          </Link>

          <Link to="/compare">
            <div className="flex items-center gap-3 p-3 rounded-lg hover:bg-emerald-50 text-gray-700 cursor-pointer transition-all">
              <GitCompare className="w-5 h-5 text-emerald-600" />
              {sidebarOpen && <span className="font-medium">Compare</span>}
            </div>
          </Link>

          <Link to="/profile">
            <div className="flex items-center gap-3 p-3 rounded-lg hover:bg-emerald-50 text-gray-700 cursor-pointer transition-all">
              <User className="w-5 h-5 text-emerald-600" />
              {sidebarOpen && <span className="font-medium">Profile</span>}
            </div>
          </Link>

          <Link to="/settings">
            <div className="flex items-center gap-3 p-3 rounded-lg hover:bg-emerald-50 text-gray-700 cursor-pointer transition-all">
              <Settings className="w-5 h-5 text-emerald-600" />
              {sidebarOpen && <span className="font-medium">Settings</span>}
            </div>
          </Link>
        </nav>
      </div>

      {/* Main Content */}
      <div 
        className={`transition-all duration-300 ${
          sidebarOpen ? 'ml-64' : 'ml-20'
        }`}
      >
        {/* Header */}
        <header className="bg-white shadow-sm border-b border-emerald-200">
          <div className="px-4 sm:px-6 lg:px-8 py-4">
            <div className="flex justify-end items-center gap-3">
              <span className="text-2xl font-bold bg-gradient-to-r from-emerald-600 to-green-600 bg-clip-text text-transparent">
                WiseBite
              </span>
              <WiseBiteLogo size={45} />
            </div>
          </div>
        </header>

        {/* Centered Scan Interface */}
        <main className="min-h-[calc(100vh-80px)] flex items-center justify-center px-4 sm:px-6 lg:px-8 py-12">
          <div className="w-full max-w-2xl">
            <Card className="border-4 border-emerald-300 shadow-2xl">
              <CardContent className="p-12">
                {/* Scanner Icon at Top */}
                <div className="flex justify-center mb-8">
                  <div className="w-24 h-24 bg-gradient-to-br from-emerald-400 to-emerald-600 rounded-full flex items-center justify-center shadow-lg">
                    <ScanLine className="w-12 h-12 text-white" />
                  </div>
                </div>

                {/* Camera Frame */}
                <div className="mb-8">
                  <div className="aspect-video bg-gradient-to-br from-emerald-50 to-green-50 rounded-3xl border-4 border-emerald-300 border-dashed flex items-center justify-center relative overflow-hidden">
                    {/* Camera Placeholder Icon */}
                    <div className="absolute inset-0 flex items-center justify-center">
                      <Camera className="w-32 h-32 text-emerald-300" />
                    </div>
                    
                    {/* Scanning Animation Lines */}
                    {isAnalyzing && (
                      <div className="absolute inset-0 flex flex-col justify-center space-y-4 px-8">
                        <div className="h-1 bg-emerald-400 rounded-full opacity-50 animate-pulse"></div>
                        <div className="h-1 bg-emerald-400 rounded-full opacity-50 animate-pulse delay-75"></div>
                        <div className="h-1 bg-emerald-400 rounded-full opacity-50 animate-pulse delay-150"></div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Message */}
                <div className="text-center mb-8">
                  <p className="text-xl text-gray-700 leading-relaxed flex flex-col items-center">
                    <span>Take a photo or upload an image of a product barcode or ingredient label.</span>
                    {selectedFile && (
                      <span className="mt-2 text-sm font-semibold text-emerald-700 bg-emerald-100 px-3 py-1 rounded-full">
                        Selected: {selectedFile.name}
                      </span>
                    )}
                  </p>
                </div>

                {/* Hidden File Input */}
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleFileSelect} 
                  accept="image/*" 
                  className="hidden" 
                />

                {/* Action Buttons */}
                <div className="space-y-4">
                  <Button 
                    size="lg" 
                    onClick={handleUploadClick}
                    className="w-full bg-gradient-to-r from-emerald-500 to-green-600 hover:from-emerald-600 hover:to-green-700 text-white rounded-xl text-lg font-semibold shadow-lg hover:shadow-xl transition-all py-6"
                    disabled={isAnalyzing}
                  >
                    <Upload className="w-6 h-6 mr-3" />
                    Upload Image
                  </Button>
                </div>

                {/* Analyze Button */}
                <div className="mt-8 pt-8 border-t-2 border-emerald-200">
                  {/* Step Status Indicator */}
                  {isAnalyzing && analysisStep && (
                    <div className="mb-4 p-3 bg-emerald-50 border-2 border-emerald-200 rounded-xl text-center">
                      <p className="text-sm font-semibold text-emerald-700 animate-pulse">{analysisStep}</p>
                    </div>
                  )}
                  <Button 
                    size="lg" 
                    onClick={handleAnalyze}
                    disabled={!selectedFile || isAnalyzing}
                    className="w-full bg-gradient-to-r from-emerald-500 via-green-500 to-emerald-500 hover:from-emerald-600 hover:via-green-600 hover:to-emerald-600 text-white rounded-full text-xl font-bold shadow-2xl py-7 disabled:opacity-50"
                  >
                    {isAnalyzing ? "Analyzing..." : "Analyze Product Image"}
                  </Button>
                  {!isAnalyzing && (
                    <p className="text-center text-xs text-gray-500 mt-3">
                      🔍 Barcode → 📦 Database → 🧪 Ingredient Label
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Helper Text */}
            <p className="text-center text-gray-600 mt-6 text-sm">
              Supported formats: JPG, PNG, HEIC • Max size: 10MB
            </p>
          </div>
        </main>
      </div>
    </div>
  );
}

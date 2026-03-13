import { useState, useRef } from "react";
import { useNavigate, Link } from "react-router";
import { Navigation } from "../components/Navigation";
import { BlossomDecoration } from "../components/BlossomDecoration";
import { Camera, Image as ImageIcon, ArrowLeft, Loader2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { extractBarcode, analyzeImage } from "../api/client";
import { healthProfile } from "../data/mockData";

export function Scanner() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loadingAction, setLoadingAction] = useState<"scan" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
    }
  };

  const handleSmartScan = async () => {
    if (!selectedImage) return;
    try {
      setLoadingAction("scan");
      setError(null);
      
      // First, attempt to extract a barcode
      try {
        const barcode = await extractBarcode(selectedImage);
        if (barcode && barcode !== "NOT_FOUND") {
          navigate(`/product/${barcode}`);
          return;
        }
      } catch (err: any) {
        // If it's a 400 Bad Request, no barcode found. Proceed to label OCR.
        console.log("No barcode found, falling back to label OCR...");
      }

      // Fallback: analyze as ingredient label
      const allergies = healthProfile.allergies;
      const diseases = healthProfile.conditions;

      const analysisRaw = await analyzeImage(selectedImage, { allergies, diseases });
      navigate(`/product/IMAGE_SCAN`, { state: { analysisRaw } });
    } catch (err: any) {
      setError(err.message ?? "Failed to analyze image.");
      setLoadingAction(null);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-50 via-purple-50 to-white relative overflow-hidden">
      <BlossomDecoration />
      <Navigation />
      
      <main className="max-w-3xl mx-auto px-4 sm:px-6 py-8 relative z-10">
        <Link to="/">
          <Button variant="ghost" className="mb-6 hover:bg-pink-50 text-pink-600 rounded-full">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Dashboard
          </Button>
        </Link>

        <Card className="border-4 border-pink-200 shadow-xl overflow-hidden">
          <CardHeader className="bg-gradient-to-r from-pink-50 to-purple-50 border-b-2 border-pink-200 flex flex-col items-center py-8">
             <div className="w-20 h-20 bg-gradient-to-br from-pink-400 to-pink-600 rounded-full flex items-center justify-center shadow-lg mb-4">
               <Camera className="w-10 h-10 text-white" />
             </div>
             <CardTitle className="text-3xl font-bold bg-gradient-to-r from-pink-600 to-purple-600 bg-clip-text text-transparent text-center">
               Scanner
             </CardTitle>
             <CardDescription className="text-center text-gray-600 text-lg mt-2">
               Take a photo or upload an image of a product's barcode or ingredient label.
             </CardDescription>
          </CardHeader>
          <CardContent className="p-8">
            <div className="flex flex-col items-center gap-6">
              
              {/* Hidden File Input */}
              <input
                type="file"
                accept="image/*"
                capture="environment"
                className="hidden"
                ref={fileInputRef}
                onChange={handleFileChange}
              />

              {previewUrl ? (
                <div className="w-full max-w-sm rounded-2xl overflow-hidden border-4 border-pink-200 shadow-md relative">
                   <img src={previewUrl} alt="Preview" className="w-full h-auto object-cover max-h-80" />
                   <Button 
                      variant="secondary" 
                      size="sm" 
                      className="absolute top-2 right-2 rounded-full opacity-80 hover:opacity-100"
                      onClick={() => { setSelectedImage(null); setPreviewUrl(null); }}
                      disabled={loadingAction !== null}
                    >
                      Clear
                   </Button>
                </div>
              ) : (
                <div 
                   onClick={() => fileInputRef.current?.click()}
                   className="w-full max-w-sm aspect-video border-4 border-dashed border-pink-300 rounded-2xl flex flex-col items-center justify-center gap-3 bg-pink-50/50 hover:bg-pink-50 cursor-pointer transition-colors"
                >
                   <ImageIcon className="w-12 h-12 text-pink-400" />
                   <span className="text-pink-600 font-medium">Capture or Upload Image</span>
                </div>
              )}

              {error && (
                <div className="w-full p-4 bg-red-50 text-red-600 rounded-xl border border-red-200 text-sm font-medium">
                  {error}
                </div>
              )}

              <div className="w-full mt-4">
                <Button
                  className="w-full py-6 text-lg bg-gradient-to-r from-pink-500 to-pink-600 hover:from-pink-600 hover:to-pink-700 shadow-md rounded-xl disabled:opacity-50"
                  disabled={!selectedImage || loadingAction !== null}
                  onClick={handleSmartScan}
                >
                  {loadingAction === "scan" ? (
                    <><Loader2 className="mr-2 h-5 w-5 animate-spin" /> Analyzing Image...</>
                  ) : (
                    "Analyze Product Image"
                  )}
                </Button>
              </div>

            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}

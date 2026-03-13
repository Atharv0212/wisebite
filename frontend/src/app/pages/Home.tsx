import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router";
import { Navigation } from "../components/Navigation";
import { BlossomDecoration } from "../components/BlossomDecoration";
import { Camera, Upload, AlertTriangle, CheckCircle, Clock, Sparkles } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Progress } from "../components/ui/progress";
import { Badge } from "../components/ui/badge";
import { healthProfile } from "../data/mockData";
import { getRecentScans } from "../utils/storage";

export function Home() {
  const navigate = useNavigate();
  const [recentScans, setRecentScans] = useState<any[]>([]);

  useEffect(() => {
    setRecentScans(getRecentScans());
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-50 via-purple-50 to-white relative overflow-hidden">
      <BlossomDecoration />
      <Navigation />
      
      {/* Decorative cherry blossom branch */}
      <div className="absolute top-0 right-0 w-96 h-96 opacity-20 pointer-events-none">
        <img 
          src="https://images.unsplash.com/photo-1651472989304-a2b2390617ca?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjaGVycnklMjBibG9zc29tJTIwc2FrdXJhJTIwcGluayUyMGZsb3dlcnN8ZW58MXx8fHwxNzczNDA1MDY0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
          alt=""
          className="w-full h-full object-cover"
        />
      </div>
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
        {/* Hero Section */}
        <div className="mb-8 text-center">
          <div className="inline-block mb-4">
            <div className="flex items-center gap-2 bg-gradient-to-r from-pink-100 to-purple-100 px-6 py-2 rounded-full border-2 border-pink-300 shadow-lg">
              <Sparkles className="w-5 h-5 text-pink-500 animate-sakura-bloom" />
              <span className="text-sm font-semibold bg-gradient-to-r from-pink-600 to-purple-600 bg-clip-text text-transparent">
                AI-Powered Health Guardian
              </span>
            </div>
          </div>
          <h1 className="text-5xl font-bold bg-gradient-to-r from-pink-600 via-purple-600 to-pink-600 bg-clip-text text-transparent mb-3">
            Welcome to WiseBite
          </h1>
          <p className="text-xl text-gray-600">
            Your journey to mindful eating begins here 🌸
          </p>
        </div>

        {/* Quick Actions */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <Card className="border-4 border-pink-200 hover:border-pink-400 transition-all cursor-pointer group overflow-hidden relative shadow-xl">
            {/* Decorative corner blossom */}
            <div className="absolute top-0 right-0 w-32 h-32 opacity-10">
              <svg viewBox="0 0 100 100" className="text-pink-400">
                <circle cx="80" cy="20" r="15" fill="currentColor"/>
                <circle cx="70" cy="10" r="10" fill="currentColor"/>
                <circle cx="90" cy="10" r="8" fill="currentColor"/>
              </svg>
            </div>
            <CardContent className="p-8 relative">
              <div className="flex items-start gap-4">
                <div className="w-20 h-20 bg-gradient-to-br from-pink-400 to-pink-600 rounded-3xl flex items-center justify-center group-hover:scale-110 transition-transform shadow-lg">
                  <Camera className="w-10 h-10 text-white" />
                </div>
                <div className="flex-1">
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">
                    Scan Barcode
                  </h3>
                  <p className="text-gray-600 mb-4 leading-relaxed">
                    Instantly analyze any product by scanning its barcode
                  </p>
                  <Link to="/scan">
                    <Button
                      className="bg-gradient-to-r from-pink-500 to-pink-600 hover:from-pink-600 hover:to-pink-700 shadow-md rounded-full"
                    >
                      Start Scanning
                    </Button>
                  </Link>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-4 border-purple-200 hover:border-purple-400 transition-all cursor-pointer group overflow-hidden relative shadow-xl">
            {/* Decorative corner blossom */}
            <div className="absolute top-0 left-0 w-32 h-32 opacity-10">
              <svg viewBox="0 0 100 100" className="text-purple-400">
                <circle cx="20" cy="20" r="15" fill="currentColor"/>
                <circle cx="30" cy="10" r="10" fill="currentColor"/>
                <circle cx="10" cy="10" r="8" fill="currentColor"/>
              </svg>
            </div>
            <CardContent className="p-8 relative">
              <div className="flex items-start gap-4">
                <div className="w-20 h-20 bg-gradient-to-br from-purple-400 to-purple-600 rounded-3xl flex items-center justify-center group-hover:scale-110 transition-transform shadow-lg">
                  <Upload className="w-10 h-10 text-white" />
                </div>
                <div className="flex-1">
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">
                    Upload Label Image
                  </h3>
                  <p className="text-gray-600 mb-4 leading-relaxed">
                    Take or upload a photo of the ingredient label
                  </p>
                  <Link to="/scan">
                    <Button className="bg-gradient-to-r from-purple-500 to-purple-600 hover:from-purple-600 hover:to-purple-700 shadow-md rounded-full">
                      Upload Photo
                    </Button>
                  </Link>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Profile Status */}
        <Card className="mb-8 border-3 border-pink-300 overflow-hidden relative shadow-xl">
          {/* Wave decoration */}
          <div className="absolute inset-0 opacity-5 pointer-events-none">
            <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
              <pattern id="home-wave" x="0" y="0" width="200" height="40" patternUnits="userSpaceOnUse">
                <path d="M0 20 Q 50 10, 100 20 T 200 20" fill="none" stroke="#ec4899" strokeWidth="2"/>
              </pattern>
              <rect width="100%" height="100%" fill="url(#home-wave)" />
            </svg>
          </div>
          <CardHeader className="bg-gradient-to-r from-pink-50 to-purple-50">
            <div className="flex items-center justify-between">
              <CardTitle className="text-2xl flex items-center gap-2">
                <span className="text-3xl">🌸</span>
                Profile Status
              </CardTitle>
              <Link to="/profile">
                <Button variant="outline" size="sm" className="border-pink-300 text-pink-600 hover:bg-pink-50 rounded-full">
                  Complete Profile
                </Button>
              </Link>
            </div>
          </CardHeader>
          <CardContent className="p-6 relative">
            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-3">
                  <span className="font-semibold text-gray-700">
                    Profile Completeness
                  </span>
                  <span className="text-2xl font-bold bg-gradient-to-r from-pink-600 to-purple-600 bg-clip-text text-transparent">
                    {healthProfile.completeness}%
                  </span>
                </div>
                <div className="relative">
                  <Progress value={healthProfile.completeness} className="h-4 bg-pink-100" />
                  <div className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-white">
                    {healthProfile.completeness >= 50 && `${healthProfile.completeness}%`}
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t-2 border-pink-100">
                <div className="bg-gradient-to-br from-pink-50 to-white p-3 rounded-xl border border-pink-200">
                  <p className="text-xs text-pink-600 font-semibold mb-1">🥜 Allergies</p>
                  <p className="text-lg font-bold text-gray-900">{healthProfile.allergies.length} tracked</p>
                </div>
                <div className="bg-gradient-to-br from-purple-50 to-white p-3 rounded-xl border border-purple-200">
                  <p className="text-xs text-purple-600 font-semibold mb-1">🍱 Diet</p>
                  <p className="text-lg font-bold text-gray-900">{healthProfile.dietaryRestrictions.length} active</p>
                </div>
                <div className="bg-gradient-to-br from-pink-50 to-white p-3 rounded-xl border border-pink-200">
                  <p className="text-xs text-pink-600 font-semibold mb-1">💊 Conditions</p>
                  <p className="text-lg font-bold text-gray-900">{healthProfile.conditions.length} monitored</p>
                </div>
                <div className="bg-gradient-to-br from-purple-50 to-white p-3 rounded-xl border border-purple-200">
                  <p className="text-xs text-purple-600 font-semibold mb-1">⭐ Preferences</p>
                  <p className="text-lg font-bold text-gray-900">{healthProfile.preferences.length} set</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Recent Scans */}
        <div>
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-3xl font-bold bg-gradient-to-r from-pink-600 to-purple-600 bg-clip-text text-transparent flex items-center gap-2">
              <span className="text-4xl">🎋</span>
              Recent Scans
            </h2>
            <Button variant="ghost" size="sm" className="text-pink-600 hover:bg-pink-50 rounded-full">
              View All →
            </Button>
          </div>
          
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5">
            {recentScans.map((product) => (
              <Link key={product.id} to={`/product/${product.id}`}>
                <Card className="hover:shadow-2xl transition-all cursor-pointer h-full border-3 border-pink-100 hover:border-pink-300 overflow-hidden group">
                  <CardContent className="p-0">
                    <div className="aspect-square bg-gradient-to-br from-pink-50 to-purple-50 overflow-hidden relative">
                      <img
                        src={product.image}
                        alt={product.name}
                        className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                      />
                      {/* Decorative corner */}
                      <div className="absolute top-2 right-2 w-8 h-8 bg-white rounded-full flex items-center justify-center shadow-md">
                        <span className="text-lg">🌸</span>
                      </div>
                    </div>
                    <div className="p-4 bg-gradient-to-br from-white to-pink-50">
                      <h3 className="font-bold text-gray-900 mb-1 line-clamp-2">
                        {product.name}
                      </h3>
                      <p className="text-sm text-gray-500 mb-3">{product.brand}</p>
                      
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-xs text-gray-500 flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {new Date(product.scanDate).toLocaleDateString()}
                        </span>
                        <div className="flex items-center gap-1">
                          <span
                            className="text-2xl font-bold"
                            style={{
                              color:
                                product.healthScore >= 80
                                  ? "#10b981"
                                  : product.healthScore >= 60
                                  ? "#f59e0b"
                                  : product.healthScore >= 40
                                  ? "#f97316"
                                  : "#ef4444",
                            }}
                          >
                            {product.healthScore}
                          </span>
                          <span className="text-xs text-gray-400">/ 100</span>
                        </div>
                      </div>

                      {product.alerts.length > 0 ? (
                        <Badge className="w-full justify-center bg-gradient-to-r from-red-400 to-pink-400 border-none text-white">
                          <AlertTriangle className="w-3 h-3 mr-1" />
                          {product.alerts.length} Alert{product.alerts.length > 1 ? 's' : ''}
                        </Badge>
                      ) : (
                        <Badge className="w-full justify-center bg-gradient-to-r from-green-400 to-emerald-400 border-none text-white">
                          <CheckCircle className="w-3 h-3 mr-1" />
                          Safe
                        </Badge>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}
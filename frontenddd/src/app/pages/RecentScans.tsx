import { Link } from "react-router";
import { WiseBiteLogo } from "../components/WiseBiteLogo";
import { AlertTriangle, CheckCircle, Clock, Home as HomeIcon, GitCompare, User, Menu, Settings, ScanLine, History, Trash2 } from "lucide-react";
import { Card, CardContent } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { useState } from "react";
import { getRecentScans, clearRecentScans, RecentScan } from "../utils/api";

export function RecentScans() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [scans, setScans] = useState<RecentScan[]>(getRecentScans());

  const handleClear = () => {
    if (confirm("Clear all recent scans?")) {
      clearRecentScans();
      setScans([]);
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
            <div className="flex items-center gap-3 p-3 rounded-lg hover:bg-emerald-50 text-gray-700 cursor-pointer transition-all">
              <ScanLine className="w-5 h-5 text-emerald-600" />
              {sidebarOpen && <span className="font-medium">Scan</span>}
            </div>
          </Link>

          <Link to="/recent-scans">
            <div className="flex items-center gap-3 p-3 rounded-lg bg-gradient-to-r from-emerald-500 to-emerald-600 text-white shadow-md cursor-pointer">
              <History className="w-5 h-5" />
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
            <div className="flex justify-between items-center">
              <h1 className="text-2xl font-bold text-gray-900">Recent Scans</h1>
              <div className="flex items-center gap-3">
                <span className="text-2xl font-bold bg-gradient-to-r from-emerald-600 to-green-600 bg-clip-text text-transparent">
                  WiseBite
                </span>
                <WiseBiteLogo size={45} />
              </div>
            </div>
          </div>
        </header>

        <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="mb-6 flex justify-between items-center">
            <p className="text-gray-600">
              {scans.length > 0
                ? `You have ${scans.length} scanned product${scans.length !== 1 ? 's' : ''}`
                : 'No scans yet. Go to Scan to analyze a product!'}
            </p>
            {scans.length > 0 && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleClear}
                className="text-red-600 border-red-300 hover:bg-red-50"
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Clear All
              </Button>
            )}
          </div>

          {/* Empty State */}
          {scans.length === 0 && (
            <Card className="border-3 border-emerald-200 shadow-xl">
              <CardContent className="p-12 text-center">
                <ScanLine className="w-16 h-16 text-emerald-300 mx-auto mb-4" />
                <h3 className="text-xl font-bold text-gray-700 mb-2">No Recent Scans</h3>
                <p className="text-gray-500 mb-6">Upload a product image to get your first analysis</p>
                <Link to="/scan">
                  <Button className="bg-gradient-to-r from-emerald-500 to-green-600 text-white px-8 py-3 rounded-xl font-semibold shadow-lg">
                    Start Scanning
                  </Button>
                </Link>
              </CardContent>
            </Card>
          )}

          {/* Scan Cards */}
          <div className="space-y-5">
            {scans.map((scan) => (
              <Link key={scan.id} to={`/product/${scan.barcode}`} state={{ analysisResult: scan.analysisResult }}>
                <Card className="border-3 border-emerald-200 hover:border-emerald-400 hover:shadow-xl transition-all cursor-pointer mb-4">
                  <CardContent className="p-6">
                    <div className="flex gap-6">
                      {/* Product Icon */}
                      <div className="w-28 h-28 flex-shrink-0 bg-gradient-to-br from-emerald-50 to-green-100 rounded-xl overflow-hidden border-2 border-emerald-200 flex items-center justify-center">
                        <span className="text-4xl">
                          {scan.healthScore >= 80 ? '🥬' : scan.healthScore >= 60 ? '🍊' : '⚠️'}
                        </span>
                      </div>

                      {/* Product Info */}
                      <div className="flex-1">
                        <div className="flex justify-between items-start mb-3">
                          <div>
                            <h3 className="text-xl font-bold text-gray-900 mb-1">
                              {scan.product_name}
                            </h3>
                            <p className="text-sm text-gray-600 capitalize">{scan.category.replace('_', ' ')}</p>
                          </div>
                          <div className="text-right">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="text-xs text-gray-500">Health Score:</span>
                              <span
                                className="text-3xl font-bold"
                                style={{
                                  color:
                                    scan.healthScore >= 80
                                      ? "#10b981"
                                      : scan.healthScore >= 60
                                      ? "#f59e0b"
                                      : scan.healthScore >= 40
                                      ? "#f97316"
                                      : "#ef4444",
                                }}
                              >
                                {scan.healthScore}
                              </span>
                            </div>
                            <span className="text-xs text-gray-400">/ 100</span>
                          </div>
                        </div>

                        <div className="grid grid-cols-3 gap-4 mb-3">
                          {/* Risk Level */}
                          <div className="bg-gradient-to-br from-emerald-50 to-white p-3 rounded-lg border border-emerald-200">
                            <p className="text-xs text-gray-600 mb-1">Risk Level</p>
                            <Badge 
                              className={`${
                                scan.healthScore >= 80
                                  ? "bg-green-500"
                                  : scan.healthScore >= 60
                                  ? "bg-yellow-500"
                                  : "bg-red-500"
                              } text-white border-none`}
                            >
                              {scan.healthScore >= 80 ? "Low" : scan.healthScore >= 60 ? "Moderate" : "High"}
                            </Badge>
                          </div>

                          {/* Hazards */}
                          <div className="bg-gradient-to-br from-white to-emerald-50 p-3 rounded-lg border border-emerald-200">
                            <p className="text-xs text-gray-600 mb-1">Hazards</p>
                            {scan.hazardCount > 0 ? (
                              <div className="flex items-center gap-1 text-red-600">
                                <AlertTriangle className="w-4 h-4" />
                                <span className="font-bold">{scan.hazardCount}</span>
                              </div>
                            ) : (
                              <div className="flex items-center gap-1 text-green-600">
                                <CheckCircle className="w-4 h-4" />
                                <span className="font-bold">None</span>
                              </div>
                            )}
                          </div>

                          {/* Safe Ingredients */}
                          <div className="bg-gradient-to-br from-emerald-50 to-white p-3 rounded-lg border border-emerald-200">
                            <p className="text-xs text-gray-600 mb-1">Safe</p>
                            <span className="font-bold text-emerald-600">{scan.safeCount} ingredients</span>
                          </div>
                        </div>

                        {/* Scan Date & Summary */}
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-gray-500 flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            Scanned {new Date(scan.scanDate).toLocaleDateString()} at {new Date(scan.scanDate).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                          </span>
                          {scan.warningCount > 0 && (
                            <Badge className="bg-orange-100 text-orange-700 border-orange-300 text-xs">
                              {scan.warningCount} warning{scan.warningCount !== 1 ? 's' : ''}
                            </Badge>
                          )}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        </main>
      </div>
    </div>
  );
}
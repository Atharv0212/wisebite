import { useState } from "react";
import { Link } from "react-router";
import { WiseBiteLogo } from "../components/WiseBiteLogo";
import { TrendingUp, AlertTriangle, CheckCircle, Sparkles, Home as HomeIcon, GitCompare, User, Menu, Settings, ScanLine, History, Leaf } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { getRecentScans, RecentScan } from "../utils/api";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "../components/ui/dialog";

export function Comparison() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const recentScans = getRecentScans();

  const [selectedA, setSelectedA] = useState<RecentScan | null>(recentScans[0] || null);
  const [selectedB, setSelectedB] = useState<RecentScan | null>(recentScans[1] || null);
  const [isDialogAOpen, setIsDialogAOpen] = useState(false);
  const [isDialogBOpen, setIsDialogBOpen] = useState(false);

  const winner = selectedA && selectedB
    ? (selectedA.healthScore > selectedB.healthScore ? selectedA : selectedB)
    : null;

  const ScanSelector = ({
    onSelect,
    excludeId,
    position,
  }: {
    onSelect: (scan: RecentScan) => void;
    excludeId: string | null;
    position: "A" | "B";
  }) => (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-h-96 overflow-y-auto">
      {recentScans
        .filter((s) => s.id !== excludeId)
        .map((scan) => (
          <Card
            key={scan.id}
            className="cursor-pointer hover:shadow-lg transition-all border-2 border-emerald-200 hover:border-emerald-400"
            onClick={() => {
              onSelect(scan);
              if (position === "A") setIsDialogAOpen(false);
              else setIsDialogBOpen(false);
            }}
          >
            <CardContent className="p-4">
              <div className="flex items-center gap-3 mb-2">
                <span className="text-2xl">
                  {scan.healthScore >= 80 ? '🥬' : scan.healthScore >= 60 ? '🍊' : '⚠️'}
                </span>
                <div className="flex-1 min-w-0">
                  <h4 className="font-semibold text-sm text-gray-900 truncate">{scan.product_name}</h4>
                  <p className="text-xs text-gray-500 capitalize">{scan.category.replace('_', ' ')}</p>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500">Score:</span>
                <span
                  className="text-lg font-bold"
                  style={{
                    color: scan.healthScore >= 80 ? "#10b981" : scan.healthScore >= 60 ? "#f59e0b" : "#ef4444",
                  }}
                >
                  {scan.healthScore}
                </span>
              </div>
              <p className="text-xs text-gray-400 mt-1">
                {new Date(scan.scanDate).toLocaleDateString()}
              </p>
            </CardContent>
          </Card>
        ))}
      {recentScans.filter((s) => s.id !== excludeId).length === 0 && (
        <p className="col-span-2 text-center text-gray-500 p-4">No other scans available. Scan more products first!</p>
      )}
    </div>
  );

  // Helper to render a product comparison card
  const renderProductCard = (scan: RecentScan | null, label: string, isWinner: boolean, colorScheme: "emerald" | "green", dialogOpen: boolean, setDialogOpen: (v: boolean) => void, onSelect: (s: RecentScan) => void, excludeId: string | null, position: "A" | "B") => (
    <Card className={`border-4 overflow-hidden shadow-xl ${isWinner ? `border-${colorScheme}-500 ring-4 ring-${colorScheme}-200` : `border-${colorScheme}-300`}`}>
      <CardHeader className={`bg-gradient-to-br from-${colorScheme}-50 via-green-50 to-${colorScheme}-50 pb-6`}>
        <div className="flex justify-between items-start mb-4">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-3">
              <Badge className={`bg-gradient-to-r from-${colorScheme}-500 to-green-500 text-white border-none text-sm px-3 py-1`}>
                {label}
              </Badge>
              {isWinner && (
                <Badge className="bg-gradient-to-r from-green-400 to-emerald-400 text-white border-none shadow-md text-sm px-3 py-1">
                  <TrendingUp className="w-3.5 h-3.5 mr-1" />
                  Winner
                </Badge>
              )}
            </div>
            {scan ? (
              <>
                <p className={`text-sm text-${colorScheme}-700 font-semibold mb-1 capitalize`}>{scan.category.replace('_', ' ')}</p>
                <CardTitle className="text-2xl text-gray-900 leading-tight">{scan.product_name}</CardTitle>
              </>
            ) : (
              <CardTitle className="text-xl text-gray-400">Select a product</CardTitle>
            )}
          </div>

          <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
            <DialogTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                className={`border-2 border-${colorScheme}-300 text-${colorScheme}-700 hover:bg-${colorScheme}-50 rounded-lg font-semibold`}
              >
                {scan ? "Change" : "Select"}
              </Button>
            </DialogTrigger>
            <DialogContent className={`max-w-3xl bg-gradient-to-br from-${colorScheme}-50 to-white border-4 border-${colorScheme}-300`}>
              <DialogHeader>
                <DialogTitle className={`text-2xl bg-gradient-to-r from-${colorScheme}-600 to-green-600 bg-clip-text text-transparent`}>
                  Select {label}
                </DialogTitle>
                <DialogDescription>
                  Choose a recently scanned product to compare
                </DialogDescription>
              </DialogHeader>
              <ScanSelector onSelect={onSelect} excludeId={excludeId} position={position} />
            </DialogContent>
          </Dialog>
        </div>

        {/* Score Display */}
        {scan && (
          <div className="bg-white rounded-2xl p-6 border-3 border-emerald-200 shadow-md text-center">
            <p className="text-sm text-gray-500 mb-1">Health Score</p>
            <span
              className="text-5xl font-bold"
              style={{
                color: scan.healthScore >= 80 ? "#10b981" : scan.healthScore >= 60 ? "#f59e0b" : scan.healthScore >= 40 ? "#f97316" : "#ef4444",
              }}
            >
              {scan.healthScore}
            </span>
            <span className="text-gray-400 text-lg"> / 100</span>
          </div>
        )}
      </CardHeader>

      {scan && (
        <CardContent className="pt-8 pb-6 bg-white space-y-5">
          {/* Hazard Summary */}
          <div className="bg-gradient-to-br from-emerald-50 to-white p-5 rounded-2xl border-2 border-emerald-200">
            <h4 className="font-bold text-gray-900 mb-3 flex items-center gap-2 text-base">
              <AlertTriangle className="w-5 h-5 text-emerald-600" />
              Risk Summary
            </h4>
            <div className="grid grid-cols-3 gap-3">
              <div className="text-center p-2 bg-red-50 rounded-xl border border-red-200">
                <p className="text-xs text-red-600 font-semibold">Hazards</p>
                <p className="text-2xl font-bold text-red-600">{scan.hazardCount}</p>
              </div>
              <div className="text-center p-2 bg-amber-50 rounded-xl border border-amber-200">
                <p className="text-xs text-amber-600 font-semibold">Cautions</p>
                <p className="text-2xl font-bold text-amber-600">{scan.cautionCount}</p>
              </div>
              <div className="text-center p-2 bg-green-50 rounded-xl border border-green-200">
                <p className="text-xs text-green-600 font-semibold">Safe</p>
                <p className="text-2xl font-bold text-green-600">{scan.safeCount}</p>
              </div>
            </div>
          </div>

          {/* Key Ingredients */}
          <div className="bg-gradient-to-br from-white to-emerald-50 p-5 rounded-2xl border-2 border-emerald-200">
            <h4 className="font-bold text-gray-900 mb-3 flex items-center gap-2 text-base">
              <Leaf className="w-5 h-5 text-green-600" />
              Key Ingredients
            </h4>
            <div className="space-y-2.5">
              {scan.analysisResult.ingredient_risks.slice(0, 5).map((ing, idx) => (
                <div key={idx} className="flex items-center gap-2.5">
                  <div
                    className={`w-3 h-3 rounded-full ${
                      ing.hazard_level === "safe"
                        ? "bg-green-400"
                        : ing.hazard_level === "caution"
                        ? "bg-amber-400"
                        : "bg-red-400"
                    }`}
                  />
                  <span className="text-sm text-gray-700 font-medium capitalize">{ing.ingredient}</span>
                </div>
              ))}
              {scan.analysisResult.ingredient_risks.length === 0 && (
                <p className="text-sm text-gray-400">No ingredient data</p>
              )}
            </div>
          </div>

          {/* Warnings */}
          {scan.warningCount > 0 && (
            <div className="bg-gradient-to-br from-orange-50 to-white p-5 rounded-2xl border-2 border-orange-200">
              <h4 className="font-bold text-gray-900 mb-3 text-base">⚠️ Warnings</h4>
              <div className="flex flex-wrap gap-2">
                {scan.analysisResult.warnings.slice(0, 2).map((w, idx) => (
                  <Badge key={idx} className="bg-orange-100 text-orange-700 border-orange-300 text-xs">
                    {w.length > 60 ? w.substring(0, 60) + '...' : w}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      )}
    </Card>
  );

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
          <Button variant="ghost" size="sm" onClick={() => setSidebarOpen(!sidebarOpen)} className="hover:bg-emerald-50">
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
            <div className="flex items-center gap-3 p-3 rounded-lg hover:bg-emerald-50 text-gray-700 cursor-pointer transition-all">
              <History className="w-5 h-5 text-emerald-600" />
              {sidebarOpen && <span className="font-medium">Recent Scans</span>}
            </div>
          </Link>
          <Link to="/compare">
            <div className="flex items-center gap-3 p-3 rounded-lg bg-gradient-to-r from-emerald-500 to-emerald-600 text-white shadow-md cursor-pointer">
              <GitCompare className="w-5 h-5" />
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
      <div className={`transition-all duration-300 ${sidebarOpen ? 'ml-64' : 'ml-20'}`}>
        {/* Header */}
        <header className="bg-white shadow-sm border-b border-emerald-200">
          <div className="px-4 sm:px-6 lg:px-8 py-5">
            <div className="flex justify-between items-center">
              <div className="inline-flex items-center gap-3">
                <div className="w-12 h-12 bg-gradient-to-br from-emerald-400 to-emerald-600 rounded-2xl flex items-center justify-center shadow-lg">
                  <GitCompare className="w-7 h-7 text-white" />
                </div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-emerald-600 via-green-600 to-emerald-600 bg-clip-text text-transparent">
                  Compare Products
                </h1>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-2xl font-bold bg-gradient-to-r from-emerald-600 to-green-600 bg-clip-text text-transparent">
                  WiseBite
                </span>
                <WiseBiteLogo size={45} />
              </div>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-8 py-12">
          {/* Hero Section */}
          <div className="mb-12 text-center">
            <div className="inline-block mb-4">
              <div className="flex items-center gap-2 bg-gradient-to-r from-emerald-100 to-green-100 px-5 py-2 rounded-full border-2 border-emerald-300 shadow-lg">
                <Sparkles className="w-5 h-5 text-emerald-600" />
                <span className="text-sm font-semibold bg-gradient-to-r from-emerald-600 to-green-600 bg-clip-text text-transparent">
                  Compare Your Recent Scans
                </span>
              </div>
            </div>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Select two scanned products for side-by-side analysis 🎋
            </p>
          </div>

          {/* Empty State */}
          {recentScans.length < 2 && (
            <Card className="border-3 border-emerald-200 shadow-xl mb-10">
              <CardContent className="p-12 text-center">
                <GitCompare className="w-16 h-16 text-emerald-300 mx-auto mb-4" />
                <h3 className="text-xl font-bold text-gray-700 mb-2">Need More Scans</h3>
                <p className="text-gray-500 mb-6">
                  You need at least 2 scanned products to compare. You currently have {recentScans.length}.
                </p>
                <Link to="/scan">
                  <Button className="bg-gradient-to-r from-emerald-500 to-green-600 text-white px-8 py-3 rounded-xl font-semibold shadow-lg">
                    Scan Products
                  </Button>
                </Link>
              </CardContent>
            </Card>
          )}

          {/* Comparison Grid */}
          {recentScans.length >= 2 && (
            <>
              <div className="grid lg:grid-cols-2 gap-8 mb-10">
                {renderProductCard(selectedA, "Product A", winner?.id === selectedA?.id, "emerald", isDialogAOpen, setIsDialogAOpen, setSelectedA, selectedB?.id || null, "A")}
                {renderProductCard(selectedB, "Product B", winner?.id === selectedB?.id, "green", isDialogBOpen, setIsDialogBOpen, setSelectedB, selectedA?.id || null, "B")}
              </div>

              {/* AI Verdict Section */}
              {selectedA && selectedB && winner && (
                <Card className="border-4 border-emerald-400 overflow-hidden shadow-2xl">
                  <CardHeader className="bg-gradient-to-r from-emerald-100 via-green-100 to-emerald-100 border-b-3 border-emerald-300 py-6">
                    <CardTitle className="flex items-center gap-3 text-3xl">
                      <div className="w-14 h-14 bg-gradient-to-br from-emerald-400 to-emerald-600 rounded-2xl flex items-center justify-center shadow-lg">
                        <Sparkles className="w-8 h-8 text-white" />
                      </div>
                      <span className="bg-gradient-to-r from-emerald-600 to-green-600 bg-clip-text text-transparent">
                        AI Verdict
                      </span>
                    </CardTitle>
                  </CardHeader>

                  <CardContent className="p-10 bg-white">
                    <div className="space-y-8">
                      {/* Recommendation Banner */}
                      <div className="bg-gradient-to-br from-emerald-50 to-green-50 rounded-3xl p-8 border-3 border-emerald-300 shadow-lg">
                        <div className="flex items-center gap-4 mb-5">
                          <span className="text-5xl">🎋</span>
                          <div>
                            <h3 className="text-3xl font-bold bg-gradient-to-r from-emerald-600 to-green-600 bg-clip-text text-transparent">
                              Recommended: {winner.product_name}
                            </h3>
                            <p className="text-gray-600 mt-1 capitalize">{winner.category.replace('_', ' ')}</p>
                          </div>
                        </div>

                        <p className="text-gray-700 text-lg leading-relaxed mb-8">
                          Based on our comprehensive analysis, <strong>{winner.product_name}</strong> is the wiser choice with a health score of <strong>{winner.healthScore}</strong> vs <strong>{winner === selectedA ? selectedB.healthScore : selectedA.healthScore}</strong>.
                        </p>

                        {/* Stats Grid */}
                        <div className="grid md:grid-cols-3 gap-6 pt-6 border-t-3 border-emerald-200">
                          <div className="text-center bg-white p-6 rounded-2xl border-2 border-emerald-300 shadow-md">
                            <p className="text-sm font-semibold text-gray-600 mb-2">Health Score Difference</p>
                            <p className="text-5xl font-bold bg-gradient-to-r from-emerald-600 to-green-600 bg-clip-text text-transparent">
                              +{Math.abs(selectedA.healthScore - selectedB.healthScore)}
                            </p>
                            <p className="text-xs text-gray-500 mt-2">points advantage</p>
                          </div>

                          <div className="text-center bg-white p-6 rounded-2xl border-2 border-green-300 shadow-md">
                            <p className="text-sm font-semibold text-gray-600 mb-2">Fewer Hazards</p>
                            <p className="text-5xl font-bold bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent">
                              {Math.abs(selectedA.hazardCount - selectedB.hazardCount)}
                            </p>
                            <p className="text-xs text-gray-500 mt-2">less dangerous ingredients</p>
                          </div>

                          <div className="text-center bg-white p-6 rounded-2xl border-2 border-emerald-300 shadow-md">
                            <p className="text-sm font-semibold text-gray-600 mb-2">Safer Ingredients</p>
                            <p className="text-5xl font-bold bg-gradient-to-r from-emerald-600 to-green-600 bg-clip-text text-transparent">
                              {winner.safeCount}
                            </p>
                            <p className="text-xs text-gray-500 mt-2">verified as safe</p>
                          </div>
                        </div>
                      </div>

                      {/* Why This Recommendation */}
                      <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-2xl p-8 border-3 border-green-300">
                        <h4 className="font-bold text-gray-900 mb-5 text-xl flex items-center gap-2">
                          <span className="text-2xl">💡</span>
                          Why this recommendation?
                        </h4>
                        <ul className="space-y-4 text-gray-700">
                          <li className="flex items-start gap-3">
                            <CheckCircle className="w-6 h-6 text-green-500 mt-0.5 flex-shrink-0" />
                            <span className="text-base">
                              Higher overall health score ({winner.healthScore} vs {winner === selectedA ? selectedB.healthScore : selectedA.healthScore})
                            </span>
                          </li>
                          <li className="flex items-start gap-3">
                            <CheckCircle className="w-6 h-6 text-green-500 mt-0.5 flex-shrink-0" />
                            <span className="text-base">
                              {winner.hazardCount === 0
                                ? "No hazardous ingredients detected"
                                : `Fewer hazardous ingredients (${winner.hazardCount} vs ${winner === selectedA ? selectedB.hazardCount : selectedA.hazardCount})`}
                            </span>
                          </li>
                          <li className="flex items-start gap-3">
                            <CheckCircle className="w-6 h-6 text-green-500 mt-0.5 flex-shrink-0" />
                            <span className="text-base">
                              {winner.warningCount <= (winner === selectedA ? selectedB.warningCount : selectedA.warningCount)
                                ? "Fewer overall warnings for your health profile"
                                : "Better ingredient transparency and analysis"}
                            </span>
                          </li>
                        </ul>
                      </div>

                      {/* Action Button */}
                      <div className="flex justify-center pt-4">
                        <Link to={`/product/${winner.barcode}`} state={{ analysisResult: winner.analysisResult }}>
                          <Button size="lg" className="bg-gradient-to-r from-emerald-500 via-green-500 to-emerald-500 hover:from-emerald-600 hover:via-green-600 hover:to-emerald-600 px-12 py-7 text-xl rounded-full shadow-2xl hover:shadow-3xl transition-all">
                            View {winner.product_name} Details 🎋
                          </Button>
                        </Link>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}

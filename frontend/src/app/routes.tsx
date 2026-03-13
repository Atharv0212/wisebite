import { createBrowserRouter } from "react-router";
import { Home } from "./pages/Home";
import { ProductAnalysis } from "./pages/ProductAnalysis";
import { Comparison } from "./pages/Comparison";
import { Profile } from "./pages/Profile";
import { Scanner } from "./pages/Scanner";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Home,
  },
  {
    path: "/product/:id",
    Component: ProductAnalysis,
  },
  {
    path: "/compare",
    Component: Comparison,
  },
  {
    path: "/profile",
    Component: Profile,
  },
  {
    path: "/scan",
    Component: Scanner,
  },
]);
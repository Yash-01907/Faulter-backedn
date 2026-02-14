# Role: Senior Digital Twin & IoT Architect
# Context: Fault Prediction Backend for a Textile Inkjet Printer

I am building a predictive maintenance backend that translates mechanical stressors into electrical "Current Signatures." 

## 1. System Vision
The user builds a formula chain in a React Flow frontend. Each node represents a physical component (Motor, Heater, Hydraulic). I need a backend "Compute Engine" that parses this graph and generates "Current Signatures" for various fault states. 

## 2. Structural Requirements (The Backend "Vibe")
- **Modular Solver Strategy:** Create a `SolverInterface`. Implement a `DAGSolver` (Directed Acyclic Graph) now, but allow me to swap in an `MCSASolver` or `MatrixSolver` later without breaking the API.
- **Node-Based Execution:** Each node from the frontend (JSON) should map to a Python class.
- **State Management:** Implement a `SystemState` object that tracks variables (Torque, Temp, Tension) across the entire graph.

## 3. Algorithmic Kernels
Please implement the following logic in the backend:
- **Topological Sorting:** Use Kahn's Algorithm to determine the order of calculations.
- **Iterative Feedback (Loops):** Implement a `while` loop logic for "Convergent Solving." If Node A affects Node B, and Node B affects Node A (e.g., Heat and Resistance), iterate until the Delta is < 0.001.
- **Signature Generation:** Implement a `SweepNode` that takes a parameter (e.g., Tension), iterates from [min] to [max], and outputs a vector of Current values. This vector is my "Stored Line."

## 4. Fault Comparison (The Pointer Logic)
- Create a service that accepts a "Live Vector" (from sensors) and compares it against the "Stored Library."
- Use **Euclidean Distance** or **Cosine Similarity** to determine which "Line" the "Pointer" is closest to.
- Calculate a **Residual Score** (Live Value - Predicted Value). If the residual exceeds a threshold, flag a "Probable Fault."

## 5. Technical Stack Preferences
- Language: Python (FastAPI).
- Math: NumPy (for vector comparison).
- Graph Logic: NetworkX (to handle the React Flow JSON parsing).

## 6. Initial Task
Start by building the `GraphEngine` class. It should take a sample React Flow JSON and print the order in which the formulas should be executed.
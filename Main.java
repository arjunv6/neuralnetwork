package neuralnetwork;
import java.util.Random;
public class Main {
	public static void main(String[] args) {
		int inputsize = 5;
		int hiddensize = 11;
		int outputsize = 2;
		NeuralNetwork nn = new NeuralNetwork(inputsize, hiddensize, outputsize);
		System.out.println(nn);
	}
}

class NeuralNetwork {
	int inputsize;
	int hiddensize;		// member variables
	int outputsize;
	Matrix weights_ih;
	Matrix weights_ho;
	
	public NeuralNetwork(int i, int h,int o) {	// constructor
		inputsize = i;
		hiddensize = h;
		outputsize = o;
		Random r = new Random(314159);
		weights_ih = new Matrix(h,i,r);
		weights_ho = new Matrix(o,h,r);
	}
	public String toString() { // method
		String nn = "Here is the input size: " + inputsize + 
				"\nHere is the hidden size: "  + hiddensize + 
				"\nHere is the output size: " + outputsize +
				"\nThe following are the weights of the Neural Network--";
		nn+= weights_ih;
		return nn;
	}
	/*public double[] forwardPass(double[] input) {
		// Takes in an input, does calculation, returns output
		
	} */
}
class Matrix {
    // all things matrices. this class can be instantiated as a matrix (and then operated upon) and also has static functions
    // for performing operations upon/between passed matrices
    double[][] M;

    public Matrix(int d1, int d2) {
        // instantiates a matrix of all zeros of the desired dimensions
        M = new double[d1][d2];
    }
    public Matrix(int d1, int d2, Random r) {
        // instantiates a matrix of desired dimensions with random entries in [-1,1]
        M = new double[d1][d2];
        for (int i=0; i<d1; i++) {
            for (int j=0; j<d2; j++) {
                M[i][j] = r.nextDouble()*2-1;
            }
        }
    }
 
    public Matrix(double[][] A) {
        this.M = A;
    }
    public Matrix(double[] v) {
        // assumes that a 1-d double array passed is a column vector (rather than row vector)
        M = new double[v.length][];
        for (int i = 0; i < v.length; i++) {
            double[] t = {v[i]};
            M[i] = t;
        }
    }
    public Matrix(Matrix mX) {
        // make a copy of an existing matrix
        double[][] X = mX.M;
        M = new double[X.length][X[0].length];
        for (int i = 0; i < M.length; i++) {
            for (int j = 0; j < M[0].length; j++) {
                M[i][j] = X[i][j];
            }
        }
    }
    public void add(Matrix mB) {
        // element-wise addition (assumes equivalent dimensions)
        double[][] B = mB.M;
        for (int i=0; i<M.length; i++) {
            for (int j=0; j<M[0].length; j++) {
                this.M[i][j] += B[i][j];
            }
        }
    }
    public void subtract(Matrix mB) {
        // element-wise addition (assumes equivalent dimensions)
        double[][] B = mB.M;
        for (int i=0; i<M.length; i++) {
            for (int j=0; j<M[0].length; j++) {
                this.M[i][j] -= B[i][j];
            }
        }
    }
    public void multiply_elementwise(Matrix mB) {
        // element-wise multiplication(assumes equivalent dimensions)
        double[][] B = mB.M;
        for (int i=0; i<M.length; i++) {
            for (int j=0; j<M[0].length; j++) {
                this.M[i][j] *= B[i][j];
            }
        }
    }
    public void divide_elementwise(Matrix mB) {
        // element-wise multiplication(assumes equivalent dimensions)
        double[][] B = mB.M;
        for (int i=0; i<M.length; i++) {
            for (int j=0; j<M[0].length; j++) {
                this.M[i][j] /= B[i][j];
            }
        }
    }
    public void multiply(Matrix mB) {
        // matrix (post)multiplication (assumes equivalent dimensions)
        double[][] B = mB.M;
        for (int i = 0; i < M.length; i++) {
            for (int j = 0; j < B[0].length; j++) {
                for (int k = 0; k < M[0].length; k++) {
                    M[i][j] += M[i][k] * B[k][j];
                }
            }
        }
    }
    public void multiply(double x) {
        // matrix (post)multiplication (assumes equivalent dimensions)
        for (int i = 0; i < M.length; i++) {
            for (int j = 0; j < M[0].length; j++) {
                M[i][j] *= x;
            }
        }
    }
 
    public boolean equalDimensions(Matrix mB) {
        return equalDimensions(this, mB);
    }
    public static Matrix transpose(Matrix mB) {
        // returns a new matrix, the transpose of the passed matrix
        double[][] B = mB.M;
        double[][] BT = new double[B[0].length][B.length];
        for (int i = 0; i < B.length; i++) {
            for (int j = 0; j < B[0].length; j++) {
                BT[j][i] = B[i][j];
            }
        }
        return new Matrix(BT);
    }
    public void square() {
        for (int i = 0; i < M.length; i++) {
            for (int j = 0; j < M[0].length; j++) {
                M[i][j] = M[i][j] * M[i][j];
            }
        }
    }
    public void sigmoid() {
        for (int i = 0; i < M.length; i++) {
            for (int j = 0; j < M[0].length; j++) {
                M[i][j] = 1 / (1 + Math.exp(-1*M[i][j]));
            }
        }
    }

    // Static matrix operations:
    public static boolean equalDimensions(Matrix mA, Matrix mB) {
        double[][] A = mA.M;
        double[][] B = mB.M;
        return (A.length == B.length) && (A[0].length == B[0].length);
    }

    public static Matrix dsigmoid(Matrix mB) {
        Matrix copy1 = new Matrix(mB);
        Matrix copy2 = new Matrix(mB);
        copy2.sigmoid();
        Matrix one = ones_like(mB);
        one.subtract(copy2);
        copy1.multiply_elementwise(one);
        return copy1;
    }

    public static Matrix ones_like(Matrix mB) {
        double[][] B = mB.M;
        double[][] ones = new double[B.length][B[0].length];
        for (int i = 0; i < B.length; i++) {
            for (int j = 0; j < B[0].length; j++) {
                ones[i][j] = 1;
            }
        }
        return new Matrix(ones);
    }

    public static double[] initialize_vector(int size, Random r) {
        double[] v = new double[size];
        for (int i=0; i<size; i++) {
            v[i] = r.nextDouble();
        }
        return v;
    }

    public static double[][] initialize_matrix(int d1, int d2, Random r) {
        double[][] M = new double[d1][d2];
        for (int i=0; i<d1; i++) {
            for (int j=0; j<d2; j++) {
                M[i][j] = r.nextDouble();
            }
        }
        return M;
    }

    public static double sum(Matrix mA) {
        double[][] A = mA.M;
        double sum = 0;
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                sum += A[i][j];
            }
        }
        return sum;
    }

    public static Matrix add(Matrix mA, Matrix mB) {
        // element wise addition (assumes equal dimensions) 
        double[][] A = mA.M;
        double[][] B = mB.M;
        double[][] R = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                R[i][j] = A[i][j] + B[i][j];
            }
        }
        return new Matrix(R);
    }

    public static Matrix subtract(Matrix mA, Matrix mB) {
        // element wise subtraction (assumes equal dimensions) 
        double[][] A = mA.M;
        double[][] B = mB.M;
        double[][] R = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                R[i][j] = A[i][j] - B[i][j];
            }
        }
        return new Matrix(R);
    }

    public static Matrix multiply(Matrix mA, double s) {
        // scalar multiplication
        double[][] A = mA.M;
        double[][] R = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                R[i][j] = s * A[i][j];
            }
        }
        return new Matrix(R);
    }

    public static Matrix multiply(Matrix mA, Matrix mB) {
        // Assumes non-jagged (rectangular) 2d arrays (matrices)
        double[][] A = mA.M;
        double[][] B = mB.M;
        double[][] R = new double[A.length][B[0].length];
        for (int i = 0; i < R.length; i++) {
            for (int j = 0; j < R[0].length; j++) {
                for (int k = 0; k < A[0].length; k++) {
                    R[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return new Matrix(R);
    }

    public static double[] addVectors(double[] a, double[] b) {
        double[] r = new double[a.length];
        for (int i = 0; i < r.length; i++) {
            r[i] = a[i] + b[i];
        }
        return r;
    }

    public String toString() {
        String s = "";
        for (int i = 0; i < M.length; i++) {
            s += "\n";
            for (int j = 0; j < M[0].length; j++) {
                s += M[i][j] + "  ";
            }
        }
        return s;
    }
}
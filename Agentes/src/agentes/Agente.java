//DOMINGUEZ RENDON MELISSA
//CABALLERO CHAVEZ YAEL JESUS
//4BM2 P1:AGENTES REACTIVOS
// Clase Agente
package agentes;

import java.util.Random;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import javax.swing.ImageIcon;
import javax.swing.JLabel;

// Clase para representar un nodo en el algoritmo de búsqueda A*
class Node {
    int x, y;
    int g; // Costo del camino desde el inicio hasta este nodo
    int h; // Heurística (distancia Manhattan hasta el nodo objetivo)
    Node parent;

    public Node(int x, int y, int g, int h, Node parent) {
        this.x = x;
        this.y = y;
        this.g = g;
        this.h = h;
        this.parent = parent;
    }

    public int getF() {
        return g + h;
    }
}

public class Agente extends Thread {
    private final String nombre;
    private int i;
    private int j;
    private int posRowB; // Declarar a nivel de clase
    private int posColB; // Declarar a nivel de clase
    private final ImageIcon icon;
    private final ImageIcon migajaIcon; // Nuevo atributo para la migaja
    private final int[][] matrix;
    private final JLabel[][] tablero;
    private JLabel casillaAnterior;
    private boolean free = true;
    private final Random aleatorio = new Random(System.currentTimeMillis());

    // Direcciones posibles para moverse: arriba, abajo, izquierda, derecha
    private final int[][] dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    public Agente(String nombre, ImageIcon icon, ImageIcon migajaIcon, int[][] matrix, JLabel[][] tablero) {
        this.nombre = nombre;
        this.icon = icon;
        this.migajaIcon = migajaIcon; // Asignación del icono de la migaja
        this.matrix = matrix;
        this.tablero = tablero;

        // Inicializar la posición del agente de manera aleatoria
        do {
            this.i = aleatorio.nextInt(matrix.length);
            this.j = aleatorio.nextInt(matrix.length);
        } while (matrix[i][j] == -1); // Repetir si la posición inicial está sobre un obstáculo

        // Colocar el icono del agente en el tablero
        tablero[i][j].setIcon(icon);
    }

    @Override
    public void run() {
        while (true) {
            casillaAnterior = tablero[i][j];
            if (free) {
                // Movimiento aleatorio del agente
                int dirRow = aleatorio.nextInt(3) - 1; // -1, 0 o 1
                int dirCol = aleatorio.nextInt(3) - 1; // -1, 0 o 1
    
                // Nueva posición del agente
                int newI = i + dirRow;
                int newJ = j + dirCol;
    
                // Verificar si la nueva posición está dentro de los límites del tablero
                if (newI >= 0 && newI < matrix.length && newJ >= 0 && newJ < matrix.length) {
                    // Verificar si la nueva posición no está sobre un obstáculo
                    if (matrix[newI][newJ] != -1 && matrix[newI][newJ] != 3) {
                        if (matrix[newI][newJ] == 1){
                            free = false;
                            moverAgente(newI, newJ, free);    
                        } else {
                            moverAgente(newI, newJ, free);
                        }
                    } else {
                        // Cambiar dirección al encontrar un obstáculo
                        dirRow = aleatorio.nextInt(3) - 1;
                        dirCol = aleatorio.nextInt(3) - 1;
                        newI = i + dirRow;
                        newJ = j + dirCol;
                        if (newI >= 0 && newI < matrix.length && newJ >= 0 && newJ < matrix.length &&
                            matrix[newI][newJ] != -1 && matrix[newI][newJ] != 3) {
                            moverAgente(newI, newJ, free);
                        }
                    }
                }
            } else {
                posBase(); // Encuentra la posición de la base
                List<Node> path = findPathToBase();
                if (path != null) {
                    // Recorrer el camino encontrado y mover el agente
                    for (int k = path.size() - 1; k >= 0; k--) {
                        Node node = path.get(k);
                        moverAgente(node.x, node.y, free);
                        try {
                            // Esperar un tiempo aleatorio antes de realizar el próximo movimiento
                            sleep(100 + aleatorio.nextInt(100));
                        } catch (InterruptedException ex) {
                            ex.printStackTrace(System.out);
                        }
                    }
                    free = true; // Una vez que llega a la base, el agente está libre nuevamente
                }
            }
        }
    }
    
    public void posBase() {
        for (int row = 0; row < matrix.length; row++) {
            for (int col = 0; col < matrix[row].length; col++) {
                if (matrix[row][col] == 3) {
                    posRowB = row;
                    posColB = col;
                    return; // Terminar la búsqueda una vez que se haya encontrado la base
                }
            }
        }
    }
    
    public void moverAgente(int newI, int newJ, boolean free) {
        // Si no está libre, entonces se está moviendo hacia la madre nodriza, no dejamos migajas
        if (free) {
            // Actualizar la posición del agente en la matriz y en el tablero
            matrix[i][j] = 0; // Desocupar la casilla anterior
            i = newI;
            j = newJ;
            matrix[i][j] = 2; // Marcar la nueva casilla ocupada por el agente
            
            // Si la casilla anterior no es la posición de la madre nodriza, dejamos migaja
            if (!(i == posRowB && j == posColB)) {
                casillaAnterior.setIcon(migajaIcon); // Dejar migaja en la casilla anterior
            }
            
            // Actualizar la posición actual del agente en el tablero
            actualizarPosicion();
        } else {
            // Si el agente no está libre, significa que está regresando a la madre nodriza, no dejamos migajas
            // Actualizar la posición del agente en la matriz y en el tablero
            matrix[i][j] = 0; // Desocupar la casilla anterior
            i = newI;
            j = newJ;
            matrix[i][j] = 2; // Marcar la nueva casilla ocupada por el agente
            
            // Actualizar la posición actual del agente en el tablero sin dejar migajas
            actualizarPosicion();
        }
    }
    
    
    public synchronized void actualizarPosicion() {
        if (free) {
            casillaAnterior.setIcon(null); // Eliminar el icono del agente de la casilla anterior
        } else {
            // Establecer la migaja en la casilla anterior si no es la posición de la base
            if (!(i == posRowB && j == posColB)) {
                casillaAnterior.setIcon(migajaIcon);
            }
        }
        tablero[i][j].setIcon(icon); // Colocar el icono del agente en la nueva casilla
        System.out.println(nombre + " in -> Row: " + i + " Col:" + j);
    }

    // Método para encontrar el camino hacia la base utilizando búsqueda A*
    private List<Node> findPathToBase() {
        List<Node> path = new ArrayList<>();
        boolean[][] visited = new boolean[matrix.length][matrix[0].length];
        PriorityQueue<Node> openList = new PriorityQueue<>(Comparator.comparingInt(Node::getF));
        openList.add(new Node(i, j, 0, heuristic(i, j, posRowB, posColB), null));

        while (!openList.isEmpty()) {
            Node current = openList.poll();
            if (current.x == posRowB && current.y == posColB) {
                // Construir el camino desde el nodo final hacia atrás
                while (current != null) {
                    path.add(current);
                    current = current.parent;
                }
                return path;
            }

            visited[current.x][current.y] = true;

            // Explorar los vecinos del nodo actual
            for (int[] dir : dirs) {
                int newX = current.x + dir[0];
                int newY = current.y + dir[1];
                if (isValid(newX, newY) && matrix[newX][newY] != -1 && !visited[newX][newY]) {
                    int g = current.g + 1;
                    int h = heuristic(newX, newY, posRowB, posColB);
                    Node neighbor = new Node(newX, newY, g, h, current);
                    openList.add(neighbor);
                }
            }
        }
        return null; // No se encontró un camino
    }

    private boolean isValid(int newX, int newY) {
        // Verificar si la nueva posición está dentro de los límites del tablero
        return newX >= 0 && newX < matrix.length && newY >= 0 && newY < matrix.length;
    }

    // Heurística (distancia Manhattan) para búsqueda A*
    private int heuristic(int x1, int y1, int x2, int y2) {
        return Math.abs(x1 - x2) + Math.abs(y1 - y2);
    }
}

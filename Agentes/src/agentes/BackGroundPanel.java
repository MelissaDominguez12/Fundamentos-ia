//DOMINGUEZ RENDON MELISSA
//CABALLERO CHAVEZ YAEL JESUS
//4BM2 P1:AGENTES REACTIVOS

package agentes;

import java.awt.Dimension;
import java.awt.Graphics;
import javax.swing.ImageIcon;
import javax.swing.JPanel;


public class BackGroundPanel extends JPanel
{
    
    ImageIcon fondo;
    
    public BackGroundPanel(ImageIcon fondo)
    {
        super();
        this.fondo = fondo;
        repaint();
    }
    
    @Override
    public void paintComponent(Graphics g)
    {
        Dimension dim = this.getSize();
        int ancho = (int) dim.getWidth();
        int alto  = (int) dim.getHeight();
        g.drawImage(fondo.getImage(), 0, 0, ancho, alto, null);
        setOpaque(false);
        super.paintComponent(g);
    }
    
}

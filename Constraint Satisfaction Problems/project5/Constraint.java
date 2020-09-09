package project5;

import java.util.ArrayList;

public interface Constraint {
    boolean check(ArrayList<Item> items, ArrayList<Bag> bags); // returns true if the constraint is satisfied, else false
}

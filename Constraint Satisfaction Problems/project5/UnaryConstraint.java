package project5;

import java.util.ArrayList;

public class UnaryConstraint implements Constraint {
    Item item; // the Item this constraint applies to
    ArrayList<Bag> bags; // the bags this constraint applies to
    boolean inclusive; // true = item must be in bags, false = item must not be in bags

    public UnaryConstraint(Item item, ArrayList<Bag> bags, boolean inclusive) {
        this.item = item;
        this.bags = bags;
        this.inclusive = inclusive;
    }

    public boolean check(ArrayList<Item> items, ArrayList<Bag> bags) {
        boolean itemInBags = false;
        for (Bag b : this.bags) {
            if (b.contents.contains(item)) {
                itemInBags = true;
                break;
            }
        }
        if (inclusive) {
            return itemInBags;
        } else {
            return !itemInBags;
        }
    }
}
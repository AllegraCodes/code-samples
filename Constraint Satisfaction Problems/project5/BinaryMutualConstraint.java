package project5;

import java.util.ArrayList;

public class BinaryMutualConstraint extends AbstractBinaryConstraint implements Constraint {
    Bag bag1, bag2; // the bags this constraint applies to

    public BinaryMutualConstraint(Item item1, Item item2, Bag bag1, Bag bag2) {
        super(item1, item2);
        this.bag1 = bag1;
        this.bag2 = bag2;
    }

    public boolean check(ArrayList<Item> items, ArrayList<Bag> bags) {
        // if either item hasn't been assigned, then this constraint is not yet broken
        if (!item1.isAssigned || !item2.isAssigned) return true;

        // see where each item is: bag1, bag2, or elsewhere
        boolean item1bag1 = false;
        boolean item1bag2 = false;
        boolean item2bag1 = false;
        boolean item2bag2 = false;
        if (bag1.contents.contains(item1)) item1bag1 = true;
        if (bag2.contents.contains(item1)) item1bag2 = true;
        if (bag1.contents.contains(item2)) item2bag1 = true;
        if (bag2.contents.contains(item2)) item2bag2 = true;

        if (item1bag1) { // item1 is in bag1, so item2 must be in bag2
            if (item2bag2) return true;
            else return false;
        }
        if (item1bag2) { // item1 is in bag2, so item2 must be in bag1
            if (item2bag1) return true;
            else return false;
        }
        if (item2bag1) { // item2 is in bag1, so item1 must be in bag2
            if (item1bag2) return true;
            else return false;
        }
        if (item2bag2) { // item2 is in bag2, so item1 must be in bag1
            if (item1bag1) return true;
            else return false;
        } else return true;
    }
}

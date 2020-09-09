package project5;

import java.util.ArrayList;

public class BagConstraint implements Constraint {
    int min, max; // the minimum/maximum number of items in each bag

    public BagConstraint() {
        this.min = 0;
        this.max = Integer.MAX_VALUE;
    }

    public boolean check(ArrayList<Item> items, ArrayList<Bag> bags) {
        // we will check for >90% fullness only if all items are assigned
        boolean done = true;
        for (Item i : items) {
            if (!i.isAssigned) {
                done = false;
                break;
            }
        }

        // check each bag
        for (Bag b : bags) {
            // check max fit limit
            int size = b.contents.size();
            if (size > max) return false;
            // check <100% fullness
            if (b.weight > b.capacity) return false;
            // check >90% fullness and min fit limit
            if (done) {
                if (size < min) return false;
                int minCapacity = (b.capacity * 9) / 10;
                if (b.weight < minCapacity) return false;
            }
        }

        return true;
    }

    public void setMin(int min) {
        this.min = min;
    }

    public void setMax(int max) {
        this.max = max;
    }
}

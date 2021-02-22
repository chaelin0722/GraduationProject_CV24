package com.example.testvideo;

import android.app.Activity;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import java.util.ArrayList;

public class UsersAdapter extends RecyclerView.Adapter<UsersAdapter.CustomViewHolder> {

    private ArrayList<TotalData> mList = null;
    private Activity context = null;


    public UsersAdapter(Activity context, ArrayList<TotalData> list) {
        this.context = context;
        this.mList = list;
    }

    class CustomViewHolder extends RecyclerView.ViewHolder {
        protected TextView id;
        protected TextView situation;
        protected TextView time;
        protected TextView longitude;
        protected TextView latitude;


        public CustomViewHolder(View view) {
            super(view);
            this.id = (TextView) view.findViewById(R.id.textView_list_id);
            this.situation = (TextView) view.findViewById(R.id.textView_list_situation);
            this.time = (TextView) view.findViewById(R.id.textView_list_time);
            this.latitude = (TextView) view.findViewById(R.id.textView_list_latitude);
            this.longitude = (TextView) view.findViewById(R.id.textView_list_longitude);
        }
    }


    @Override
    public CustomViewHolder onCreateViewHolder(ViewGroup viewGroup, int viewType) {
        View view = LayoutInflater.from(viewGroup.getContext()).inflate(R.layout.item_list, null);
        CustomViewHolder viewHolder = new CustomViewHolder(view);

        return viewHolder;
    }

    @Override
    public void onBindViewHolder(@NonNull CustomViewHolder viewholder, int position) {

        viewholder.id.setText(mList.get(position).getMember_id());
        viewholder.situation.setText(mList.get(position).getMember_situation());
        viewholder.time.setText(mList.get(position).getMember_time());
        viewholder.latitude.setText(mList.get(position).getMember_latitude());
        viewholder.longitude.setText(mList.get(position).getMember_longitude());
    }

    @Override
    public int getItemCount() {
        return (null != mList ? mList.size() : 0);
    }

}